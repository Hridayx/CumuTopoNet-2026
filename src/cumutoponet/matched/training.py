"""Deterministic batch sampling and recoverable training; never computes test scores."""
from __future__ import annotations
from contextlib import contextmanager
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from torch.nn import functional as F
from .common import (IntegrityError, atomic_target,
                     read_json, write_json, workspace)
from .features import normalize, temporal_features
from .models import build_model, parameter_count, supcon_loss
from .metrics import classification_metrics
from .data import save_array, load_arrays


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def balanced_order(labels, seed, epoch):
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    lookup = dict(zip(classes, counts))
    weights = np.array([1/lookup[c] for c in labels], dtype=np.float64)
    weights /= weights.sum()
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 1729]))
    return rng.choice(len(labels), size=len(labels), replace=True, p=weights)


def select_arrays(arrays, mask):
    n = len(arrays['labels'])
    return {k: v[mask] if isinstance(v, np.ndarray) and len(v) == n else v for k, v in arrays.items()}


def training_arrays(arrays):
    return select_arrays(arrays, arrays['splits'] != 'test')


def make_inputs(arrays, indices, device, phase_seed=None):
    z = normalize(arrays['raw'][indices])
    if phase_seed is not None:
        angles = np.random.default_rng(phase_seed).uniform(-np.pi, np.pi, len(z))
        z *= np.exp(1j*angles[:, None])
    scaler = arrays['scaler']
    hoc = (arrays['hoc'][indices]-np.asarray(scaler['mean']))/np.asarray(scaler['scale'])
    values = {'iq': np.stack([z.real, z.imag], axis=1).astype(np.float32),
              'temporal': temporal_features(z), 'hoc': hoc.astype(np.float32),
              'tda': np.asarray(arrays['pi'][indices], dtype=np.float32)}
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in values.items()}


@torch.inference_mode()
def predict(model, arrays, indices, device, batch_size=256):
    model.eval()
    pieces = []
    for start in range(0, len(indices), batch_size):
        logits, _ = model(make_inputs(arrays, indices[start:start+batch_size], device))
        prob = torch.softmax(logits.float(), dim=1).cpu().numpy()
        if not np.isfinite(prob).all():
            raise FloatingPointError('Nonfinite evaluation probabilities.')
        pieces.append(prob)
    return np.concatenate(pieces) if pieces else np.empty((0, 7), dtype=np.float32)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise IntegrityError('A CUDA run cannot resume as a CPU run.')
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def _gradients_are_finite(parameters):
    """Check unscaled gradients without converting an overflow into an update.

    GradScaler records an FP16 overflow during ``unscale_`` so that ``step`` can
    skip the optimizer update and ``update`` can lower the scale.  Calling
    ``clip_grad_norm_`` with ``error_if_nonfinite=True`` before that recovery
    path turns this ordinary calibration event into a terminal run failure.
    """
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        raise FloatingPointError('No gradients were produced for the optimizer step.')
    return bool(torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all())


def scaled_optimizer_step(optimizer, scaler, parameters, max_norm=5.):
    """Clip finite gradients or let GradScaler safely recover an AMP overflow.

    Returns ``True`` only when GradScaler skipped the update and reduced its
    loss scale.  Callers must restore the pre-batch RNG state and retry that
    same batch; they must never advance the sampler after an overflow.
    """
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    scaler.unscale_(optimizer)
    if not _gradients_are_finite(parameters):
        if not scaler.is_enabled():
            raise FloatingPointError('Nonfinite unscaled gradients with AMP disabled.')
        old_scale = float(scaler.get_scale())
        # GradScaler detects the nonfinite gradients during unscale_ and skips
        # this optimizer step.  Do not clip Inf gradients: Inf * 0 can create
        # NaNs and would make the skipped update unauditable.
        scaler.step(optimizer)
        scaler.update()
        if float(scaler.get_scale()) >= old_scale:
            raise FloatingPointError('AMP overflow did not reduce the GradScaler scale.')
        return True
    torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return False


def save_checkpoint(path, state):
    with atomic_target(path) as f:
        torch.save(state, f)


@contextmanager
def termination_flag():
    flag = {'stop': False, 'signal': None}
    previous = {}
    def handler(signum, _frame):
        flag.update(stop=True, signal=signum)
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGUSR1]:
        previous[sig] = signal.signal(sig, handler)
    try:
        yield flag
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)


def train_run(spec, arrays, output, device='cuda:0', stop_after_steps=None, deadline=None):
    """Resume only this package's trusted checkpoints; refuses scientific config changes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if 'test' in arrays['splits']:
        raise IntegrityError('Training accepts train/validation arrays only; remove the held-out test partition.')
    if spec['feature_key'] != arrays['feature_key'] or spec['data_plan_id'] != arrays['data_plan_id']:
        raise IntegrityError('Training data does not match the queued specification.')
    if (output/'result.json').exists():
        result = read_json(output/'result.json')
        if result.get('spec') != spec:
            raise IntegrityError('Completed run has a different specification.')
        if not (output/'best.pt').is_file():
            raise IntegrityError('Completed run is missing its checkpoint.')
        torch.load(output/'best.pt', map_location='cpu', weights_only=False)
        return result
    device = torch.device(device)
    seed_everything(spec['seed'])
    config = spec['training']
    model = build_model(spec['model'], spec.get('width', 1.)).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=spec['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_epochs'])
    amp = bool(config['amp'] and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    train_idx = np.flatnonzero(arrays['splits'] == 'train')
    val_idx = np.flatnonzero(arrays['splits'] == 'validation')
    if not len(train_idx) or not len(val_idx):
        raise IntegrityError('Both train and validation sets are required.')
    for idx in [train_idx, val_idx]:
        if set(arrays['labels'][idx]) != set(range(7)):
            raise IntegrityError('Training/validation lacks seven-class support.')
    epoch, step, global_step, best, stale = 0, 0, 0, -1., 0
    order, history, loss_sum, examples = None, [], 0., 0
    elapsed_before = 0.
    amp_overflow_retries, amp_overflows = 0, 0
    if (output/'last.pt').exists():
        state = torch.load(output/'last.pt', map_location=device, weights_only=False)
        if state.get('spec') != spec:
            raise IntegrityError('Checkpoint specification differs; resume refused.')
        if state['device_type'] != device.type:
            raise IntegrityError('Resume device type differs.')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['amp_scaler'])
        restore_rng(state['rng'])
        epoch, step, global_step = state['epoch'], state['step'], state['global_step']
        best, stale, order, history = state['best'], state['stale'], state['sampler_order'], state['history']
        loss_sum, examples = state['epoch_loss_sum'], state['epoch_examples']
        elapsed_before = state['elapsed_seconds']
        amp_overflow_retries = int(state.get('amp_overflow_retries', 0))
        amp_overflows = int(state.get('amp_overflows', 0))
    start_time = time.monotonic()
    def snapshot():
        return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'amp_scaler': scaler.state_dict(), 'rng': rng_state(),
                'epoch': epoch, 'step': step, 'global_step': global_step, 'sampler_order': order,
                'epoch_loss_sum': loss_sum, 'epoch_examples': examples, 'best': best, 'stale': stale,
                'history': history, 'spec': spec,
                'device_type': device.type, 'elapsed_seconds': elapsed_before+time.monotonic()-start_time,
                'amp_overflow_retries': amp_overflow_retries, 'amp_overflows': amp_overflows}
    with termination_flag() as termination:
        while epoch < config['max_epochs'] and not (epoch >= config['min_epochs'] and stale >= config['patience']):
            if order is None:
                order = train_idx[balanced_order(arrays['labels'][train_idx], spec['seed'], epoch)]
            model.train()
            while step < len(order):
                if termination['stop'] or (deadline is not None and time.time() >= deadline):
                    save_checkpoint(output/'last.pt', snapshot())
                    return {'status': 'interrupted', 'signal': termination['signal'],
                            'budget_stop': deadline is not None and time.time() >= deadline}
                indices = order[step:step+config['batch_size']]
                phase_seed = np.random.SeedSequence([spec['seed'], epoch, step, 2718]) if config['phase_augmentation'] else None
                batch_rng = rng_state()
                inputs = make_inputs(arrays, indices, device, phase_seed)
                labels = torch.as_tensor(arrays['labels'][indices], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits, embedding = model(inputs)
                loss = F.cross_entropy(logits.float(), labels)
                if spec.get('supcon_weight', 0):
                    loss = loss + spec['supcon_weight']*supcon_loss(embedding, labels, config['temperature'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss.')
                scaler.scale(loss).backward()
                if scaled_optimizer_step(optimizer, scaler, model.parameters()):
                    amp_overflow_retries += 1
                    amp_overflows += 1
                    if amp_overflow_retries > 8:
                        raise FloatingPointError('AMP overflow persisted after eight identical-batch retries.')
                    # Dropout consumes CUDA RNG state during the failed forward
                    # pass.  Restore it before retrying so the recovered step is
                    # exactly the same logical minibatch.
                    restore_rng(batch_rng)
                    save_checkpoint(output/'last.pt', snapshot())
                    continue
                amp_overflow_retries = 0
                loss_sum += float(loss.detach())*len(indices)
                examples += len(indices)
                step += len(indices)
                global_step += 1
                if global_step % config['checkpoint_steps'] == 0:
                    save_checkpoint(output/'last.pt', snapshot())
                if stop_after_steps is not None and global_step >= stop_after_steps:
                    save_checkpoint(output/'last.pt', snapshot())
                    return {'status': 'interrupted', 'signal': None, 'budget_stop': False}
            probabilities = predict(model, arrays, val_idx, device, config['batch_size'])
            score = classification_metrics(arrays['labels'][val_idx], probabilities.argmax(axis=1))['macro_f1']
            history.append({'epoch': epoch+1, 'train_loss': loss_sum/max(examples, 1),
                            'validation_macro_f1': score, 'lr': optimizer.param_groups[0]['lr']})
            improved = score > best
            if improved:
                best, stale = score, 0
            else:
                stale += 1
            scheduler.step()
            epoch += 1
            step, order, loss_sum, examples = 0, None, 0., 0
            if improved:
                save_checkpoint(output/'best.pt', snapshot())
            save_checkpoint(output/'last.pt', snapshot())
            write_json(output/'progress.json', {'epoch': epoch, 'global_step': global_step,
                       'best_validation_macro_f1': best, 'elapsed_seconds': snapshot()['elapsed_seconds']})
    best_state = torch.load(output/'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(best_state['model'])
    probabilities = predict(model, arrays, val_idx, device, config['batch_size'])
    save_array(output/'validation_probabilities.npy', probabilities)
    save_array(output/'validation_ids.npy', arrays['ids'][val_idx])
    result = {'status': 'completed', 'spec': spec, 'best_validation_macro_f1': best,
              'amp_overflows': amp_overflows,
              'history': history, 'epochs': epoch, 'global_steps': global_step,
              'elapsed_seconds': elapsed_before+time.monotonic()-start_time,
              'parameters': parameter_count(model), 'device': str(device),
              'files': ['best.pt', 'validation_probabilities.npy', 'validation_ids.npy']}
    write_json(output/'result.json', result)
    return result


def numerical_smoke(cfg, device='cuda:0', steps=160, label='primary', model_name='full',
                    lr=.0003, length=128, points=None, channels=None):
    """Exercise real prepared training data before queueing scientific fits.

    This writes only a diagnostic record: it does not use test data, emit model
    predictions, or contribute a result to the experiment.  The fixed 160
    updates intentionally extend beyond the earlier real-data overflow point.
    """
    if steps < 1:
        raise ValueError('Numerical smoke requires at least one update.')
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise IntegrityError('Production numerical smoke requires an allocated CUDA device.')
    arrays = training_arrays(load_arrays(cfg, length, points, channels))
    train_idx = np.flatnonzero(arrays['splits'] == 'train')
    if not len(train_idx) or set(arrays['labels'][train_idx]) != set(range(7)):
        raise IntegrityError('Numerical smoke requires seven-class training support.')
    config = cfg['training']
    seed = 17
    seed_everything(seed)
    model = build_model(model_name).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=config['weight_decay'])
    amp = bool(config['amp'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    order = train_idx[balanced_order(arrays['labels'][train_idx], seed, 0)]
    position, completed, amp_overflows, consecutive = 0, 0, 0, 0
    started = time.monotonic()
    while completed < steps and position < len(order):
        indices = order[position:position+config['batch_size']]
        batch_rng = rng_state()
        inputs = make_inputs(arrays, indices, device, np.random.SeedSequence([seed, 0, position, 2718]))
        labels = torch.as_tensor(arrays['labels'][indices], dtype=torch.long, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
            logits, _ = model(inputs)
        loss = F.cross_entropy(logits.float(), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite real-data numerical-smoke loss.')
        scaler.scale(loss).backward()
        if scaled_optimizer_step(optimizer, scaler, model.parameters()):
            consecutive += 1
            amp_overflows += 1
            if consecutive > 8:
                raise FloatingPointError('AMP overflow persisted during real-data numerical smoke.')
            restore_rng(batch_rng)
            continue
        consecutive = 0
        position += len(indices)
        completed += 1
    if completed != min(steps, int(np.ceil(len(order)/config['batch_size']))):
        raise IntegrityError('Numerical smoke did not cover its required updates.')
    torch.cuda.synchronize(device)
    spec = arrays['feature_spec']
    record = {
        'label': str(label), 'model': model_name, 'lr': float(lr), 'steps': completed,
        'amp_overflows': amp_overflows, 'amp_final_scale': float(scaler.get_scale()),
        'config_path': cfg['_config_path'], 'data_plan_id': arrays['data_plan_id'],
        'feature_key': arrays['feature_key'],
        'feature_spec': spec, 'device': str(device), 'elapsed_seconds': time.monotonic()-started,
    }
    write_json(workspace(cfg)/'numerical_smoke'/f'{label}.json', record)
    return record
