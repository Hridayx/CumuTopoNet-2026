"""Recording inventory, conservative partitions, streaming and verified shards."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from contextlib import contextmanager
import csv
import itertools
import multiprocessing
from pathlib import Path, PurePosixPath
import re
import time
import uuid
import zipfile
import numpy as np
from .common import (IntegrityError, write_json, read_json, write_once,
                     atomic_target, directory_lock, workspace)
from .features import HOC_NAMES, feature_key, feature_spec, extract_features, fit_scaler

LABELS = ['Air2S', 'Inspire2', 'MavicMini', 'MavicPro', 'MavicPro2', 'Phantom4', 'ParrotDisco']
DRONES = dict(zip(['AIR', 'INS', 'MIN', 'MP1', 'MP2', 'PHA', 'DIS'], LABELS))
MODES = {'FY': 'Flying', 'HO': 'Hovering', 'ON': 'SwitchedOn'}
CONDITIONS = {'CLEAN': 'Clean', 'BLUE': 'BT', 'WIFI': 'WiFi', 'BOTH': 'BTandWiFi'}


def parse_recording(relative_path, cfg):
    ds = cfg['dataset']
    drones = {**DRONES, **ds.get('class_aliases', {})}
    modes = {**MODES, **ds.get('mode_aliases', {})}
    conditions = {**CONDITIONS, **ds.get('condition_aliases', {})}
    parts = PurePosixPath(relative_path).parts[:-1]
    found = []
    condition = None
    for part in parts:
        upper = part.upper()
        if upper in conditions:
            condition = conditions[upper]
        for token, label in drones.items():
            for mode_token, mode in modes.items():
                if upper in (f'{token}_{mode_token}', f'{token}-{mode_token}'):
                    found.append((label, mode))
    if len(set(found)) != 1 or condition is None:
        raise IntegrityError(f'Unrecognized recording path {relative_path!r}; configure explicit aliases.')
    label_name, mode = found[0]
    if label_name not in LABELS or mode not in MODES.values() or condition not in CONDITIONS.values():
        raise IntegrityError(f'Alias maps outside the seven-class protocol: {relative_path}')
    return {'label': LABELS.index(label_name), 'label_name': label_name,
            'mode': mode, 'condition': condition, 'family': f'{label_name}/{mode}/{condition}'}


def inventory(cfg):
    source = Path(cfg['dataset']['path'])
    entries = []
    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as z:
            seen = set()
            for info in sorted(z.infolist(), key=lambda i: i.filename):
                if info.is_dir() or not info.filename.lower().endswith('.dat') or '__MACOSX' in info.filename:
                    continue
                if info.filename in seen or '..' in PurePosixPath(info.filename).parts:
                    raise IntegrityError(f'Duplicate or unsafe archive member: {info.filename}')
                seen.add(info.filename)
                if info.flag_bits & 1:
                    raise IntegrityError('Encrypted dataset members are unsupported.')
                entries.append((info.filename, info.file_size,
                                {'size': info.file_size, 'compressed': info.compress_size}))
    elif source.is_dir():
        for p in sorted(source.rglob('*')):
            if p.is_file() and p.suffix.lower() == '.dat' and '__MACOSX' not in p.parts:
                st = p.stat()
                entries.append((p.relative_to(source).as_posix(), st.st_size,
                                {'size': st.st_size, 'mtime_ns': st.st_mtime_ns}))
    else:
        raise IntegrityError(f'Dataset is neither a readable ZIP nor directory: {source}')
    records, errors = [], []
    for rel, size, source_metadata in entries:
        try:
            record = parse_recording(rel, cfg)
            if size % 8 or size < 1024*8:
                raise IntegrityError(f'{rel}: expected interleaved little-endian float32 I/Q, got {size} bytes.')
            index = len(records)
            record.update(id=f'record-{index:04d}', index=index, path=rel, bytes=size,
                          samples=size//8, source_metadata=source_metadata)
            records.append(record)
        except IntegrityError as e:
            errors.append(str(e))
    if errors:
        raise IntegrityError('\n'.join(errors[:30]) + f'\nTotal invalid recordings: {len(errors)}')
    if {r['label'] for r in records} != set(range(7)):
        raise IntegrityError('All seven classes must be present; no implicit background classes.')
    if len({r['id'] for r in records}) != len(records):
        raise IntegrityError('Recording identifier collision.')
    # Session metadata can coarsen family grouping, never split a family.
    parent = {r['family']: r['family'] for r in records}
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    session_path = cfg['dataset'].get('session_metadata')
    if session_path:
        with open(session_path, newline='') as f:
            rows = list(csv.DictReader(f))
        sessions = {r['relative_path']: r['session_id'] for r in rows}
        if len(sessions) != len(rows) or set(sessions) != {r['path'] for r in records} or not all(sessions.values()):
            raise IntegrityError('Session CSV must have one nonempty session_id per relative_path, exactly covering inventory.')
        first_family = {}
        for r in records:
            session = sessions[r['path']]
            if session in first_family:
                parent[find(r['family'])] = find(first_family[session])
            else:
                first_family[session] = r['family']
    members = {}
    for family in parent:
        members.setdefault(find(family), []).append(family)
    group_for_family = {}
    for index, families in enumerate(sorted((sorted(values) for values in members.values()))):
        for family in families:
            group_for_family[family] = f'group-{index:04d}'
    for r in records:
        r['group'] = group_for_family[r['family']]
    observed = {r['family'] for r in records}
    possible = ['/'.join(t) for t in itertools.product(LABELS, MODES.values(), CONDITIONS.values())]
    result = {'records': records, 'labels': LABELS, 'missing_combinations': sorted(set(possible)-observed),
              'counts': {label: sum(r['label_name'] == label for r in records) for label in LABELS},
              'total_bytes': sum(r['bytes'] for r in records), 'source': str(source),
              'session_metadata': str(session_path) if session_path else None,
              'grouping': 'family-session-components'}
    return result


def make_partitions(records, seed=42):
    groups = {}
    for r in records:
        groups.setdefault(r['group'], []).append(r)
    rng = np.random.default_rng(seed)
    assignments = {}
    if all(len({r['label'] for r in rs}) == 1 for rs in groups.values()):
        totals = np.zeros(5, dtype=int)
        for label in range(7):
            keys = sorted(g for g, rs in groups.items() if rs[0]['label'] == label)
            if len(keys) < 5:
                raise IntegrityError(f'{LABELS[label]} has fewer than five independent groups.')
            rng.shuffle(keys)
            class_totals = np.zeros(5, dtype=int)
            # Large groups first; shuffled ordering breaks ties reproducibly.
            keys.sort(key=lambda g: -len(groups[g]))
            for g in keys:
                choices = np.flatnonzero(class_totals == class_totals.min())
                choices = choices[totals[choices] == totals[choices].min()]
                fold = int(rng.choice(choices))
                assignments[g] = fold
                class_totals[fold] += len(groups[g])
                totals[fold] += len(groups[g])
    else:
        from sklearn.model_selection import StratifiedGroupKFold
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (_, idx) in enumerate(splitter.split(np.zeros(len(records)),
                [r['label'] for r in records], [r['group'] for r in records])):
            for i in idx:
                assignments[records[i]['group']] = fold
    split = {r['id']: assignments[r['group']] for r in records}
    for fold in range(5):
        if {r['label'] for r in records if split[r['id']] == fold} != set(range(7)):
            raise IntegrityError(f'Partition {fold} lacks class coverage; resolve metadata/design without test-score access.')
    return split


def sample_offsets(samples, count, length=1024, seed=42):
    if count < 1 or samples < count*length:
        raise IntegrityError(f'Cannot draw {count} non-overlapping windows of length {length} from {samples}.')
    boundaries = np.linspace(0, samples, count+1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.array([rng.integers(int(a), int(b-length)+1) for a, b in
                     zip(boundaries[:-1], boundaries[1:])], dtype=np.int64)


@contextmanager
def open_recording(source, record):
    p = Path(source)
    if p.is_dir():
        with (p / record['path']).open('rb') as f:
            yield f
    else:
        with zipfile.ZipFile(p) as z:
            with z.open(record['path']) as f:
                yield f


def read_selected(source, record, offsets, length=1024):
    offsets = np.asarray(offsets, dtype=np.int64)
    if (len(offsets) == 0 or offsets[0] < 0 or np.any(np.diff(offsets) < length)
            or offsets[-1] + length > record['samples']):
        raise IntegrityError('Invalid or overlapping retained windows.')
    result = np.empty((len(offsets), length), dtype='<c8')
    position = 0
    copied = 0
    with open_recording(source, record) as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            if len(chunk) % 8:
                raise IntegrityError('Partial I/Q sample in source stream.')
            values = np.frombuffer(chunk, dtype='<c8')
            end = position + len(values)
            lo = np.searchsorted(offsets + length, position, side='right')
            hi = np.searchsorted(offsets, end, side='left')
            for i in range(lo, hi):
                a, b = max(position, int(offsets[i])), min(end, int(offsets[i])+length)
                result[i, a-offsets[i]:b-offsets[i]] = values[a-position:b-position]
                copied += b-a
            position = end
    if position != record['samples'] or copied != result.size:
        raise IntegrityError(f'Truncated or changed source: {record["path"]}')
    if not np.isfinite(result).all():
        raise IntegrityError(f'Nonfinite retained I/Q samples: {record["path"]}')
    return result


def save_array(path, array):
    with atomic_target(path) as f:
        np.save(f, array, allow_pickle=False)


def verify_shard(directory):
    directory = Path(directory)
    meta = read_json(directory/'meta.json')
    for filename in meta['files']:
        p = directory/filename
        if not p.is_file():
            raise IntegrityError(f'Missing cache file: {p}')
        if p.suffix == '.npy':
            try:
                np.load(p, mmap_mode='r', allow_pickle=False)
            except (OSError, ValueError) as error:
                raise IntegrityError(f'Unreadable cache file: {p}') from error
    return meta


def raw_shard(root, source, record, count, seed):
    directory = Path(root)/'cache'/'raw'/record['id']
    request = {'record': record, 'count': count, 'seed': seed, 'parent_length': 1024}
    if (directory/'meta.json').exists():
        meta = verify_shard(directory)
        if meta['request'] != request:
            raise IntegrityError('Raw cache request differs; use a new workspace.')
        return directory
    record_seed = int(np.random.SeedSequence([int(seed), int(record['index'])]).generate_state(1)[0])
    offsets = sample_offsets(record['samples'], count, 1024, seed=record_seed)
    raw = read_selected(source, record, offsets, 1024)
    ids = np.array([f'{record["id"]}:{int(offset)}:1024' for offset in offsets], dtype='U64')
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in [('parent', raw), ('offsets', offsets), ('ids', ids)]:
        save_array(directory/f'{name}.npy', value)
    write_json(directory/'meta.json', {'request': request,
               'files': [f'{name}.npy' for name in ['parent', 'offsets', 'ids']]})
    return directory


def feature_shard(root, raw_directory, spec):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        raw_directory = Path(raw_directory)
        raw_meta = verify_shard(raw_directory)
        directory = Path(root)/'cache'/feature_key(spec)/raw_directory.name
        request = {'feature_spec': spec, 'raw_request': raw_meta['request']}
        if (directory/'meta.json').exists():
            if verify_shard(directory)['request'] != request:
                raise IntegrityError('Feature cache is incompatible.')
            return str(directory)
        parents = np.load(raw_directory/'parent.npy', mmap_mode='r', allow_pickle=False)
        n = spec['length']
        if n not in (128, 1024):
            raise ValueError('Supported observation lengths: 128 and 1024.')
        start = (1024-n)//2
        hoc, pi = extract_features(parents[:, start:start+n], spec)
        ids = np.load(raw_directory/'ids.npy', allow_pickle=False)
        directory.mkdir(parents=True, exist_ok=True)
        for name, value in [('hoc', hoc), ('pi', pi), ('ids', ids)]:
            save_array(directory/f'{name}.npy', value)
        write_json(directory/'meta.json', {'request': request,
                   'files': [f'{name}.npy' for name in ['hoc', 'pi', 'ids']]})
        return str(directory)


def prepare_dataset(cfg, length=128, points=None, channels=None):
    root = workspace(cfg)
    with directory_lock(root/'.prepare.lock'):
        inv = inventory(cfg)
        count = cfg['dataset']['windows_per_recording']
        if (root/'budget_decision.json').exists():
            decision = read_json(root/'budget_decision.json')
            if not decision.get('feasible', True):
                raise IntegrityError('Measured budget is infeasible; resolve the resource/time constraint before preparing.')
            count = decision['windows_per_recording']
        elif not cfg.get('fixture', False):
            raise IntegrityError('Run CPU/GPU benchmarks and plan --budget before production preparation.')
        partitions = make_partitions(inv['records'])
        plan_values = {'inventory': inv, 'windows_per_recording': count, 'partitions': partitions,
                       'sample_seed': cfg['dataset']['sample_seed'], 'parent_length': 1024,
                       'split_seed': 42, 'test_fold': 0, 'validation_fold': 1}
        plan_path = root/'data_plan.json'
        if plan_path.exists():
            plan = read_json(plan_path)
            if {key: plan[key] for key in plan_values} != plan_values:
                raise IntegrityError('Prepared data settings differ; use an empty workspace.')
        else:
            plan = {**plan_values, 'plan_id': uuid.uuid4().hex}
            write_json(plan_path, plan)
        spec = feature_spec(cfg, length, points, channels)
        started = time.monotonic()
        readers = cfg['runtime']['archive_readers']
        workers = cfg['runtime']['feature_workers']
        if readers < 1 or workers < 1:
            raise ValueError('At least one reader and one feature worker are required.')
        pending = set()
        # Readers persist parents immediately. At most 2*workers feature tasks in flight.
        with ThreadPoolExecutor(readers) as reader_pool, ProcessPoolExecutor(
                workers, mp_context=multiprocessing.get_context('spawn')) as feature_pool:
            futures = [reader_pool.submit(raw_shard, root, inv['source'], r, count, plan['sample_seed'])
                       for r in inv['records']]
            for future in as_completed(futures):
                raw_directory = future.result()
                pending.add(feature_pool.submit(feature_shard, root, raw_directory, spec))
                if len(pending) >= 2*workers:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for result in done:
                        result.result()
            for future in as_completed(pending):
                future.result()
        cache_key = feature_key(spec)
        marker = {'data_plan_id': plan['plan_id'], 'feature_spec': spec,
                  'recordings': len(inv['records']), 'windows': count*len(inv['records'])}
        write_once(root/'cache'/cache_key/'complete.json', marker)
        arrays = load_arrays(cfg, length, points, channels, fit_missing_scaler=True)
        write_json(root/'preparation_status.json', {'status': 'completed', 'feature_key': cache_key,
                   'elapsed_seconds': time.monotonic()-started, 'windows': len(arrays['ids']),
                   'completed_at': time.time()})
        return marker


def load_arrays(cfg, length=128, points=None, channels=None, fit_missing_scaler=False):
    root = workspace(cfg)
    plan = read_json(root/'data_plan.json')
    spec = feature_spec(cfg, length, points, channels)
    cache_key = feature_key(spec)
    marker = read_json(root/'cache'/cache_key/'complete.json')
    if marker['data_plan_id'] != plan['plan_id'] or marker['feature_spec'] != spec:
        raise IntegrityError('Dataset/feature manifest mismatch.')
    pieces = {k: [] for k in ['raw', 'hoc', 'pi', 'ids', 'labels', 'groups', 'families',
                              'records', 'modes', 'conditions', 'splits']}
    for r in plan['inventory']['records']:
        raw_dir = root/'cache'/'raw'/r['id']
        raw_meta = verify_shard(raw_dir)
        directory = root/'cache'/cache_key/r['id']
        meta = verify_shard(directory)
        if meta['request']['raw_request'] != raw_meta['request']:
            raise IntegrityError('Feature cache does not match its raw input request.')
        ids = np.load(raw_dir/'ids.npy', allow_pickle=False)
        if not np.array_equal(ids, np.load(directory/'ids.npy', allow_pickle=False)):
            raise IntegrityError('Feature window IDs are not aligned.')
        raw = np.load(raw_dir/'parent.npy', mmap_mode='r', allow_pickle=False)
        start = (1024-length)//2
        pieces['raw'].append(np.array(raw[:, start:start+length]))
        pieces['ids'].append(ids)
        for key in ['hoc', 'pi']:
            pieces[key].append(np.load(directory/f'{key}.npy', allow_pickle=False))
        split = {0: 'test', 1: 'validation'}.get(plan['partitions'][r['id']], 'train')
        for key, val in [('labels', r['label']), ('groups', r['group']), ('families', r['family']),
                         ('records', r['id']), ('modes', r['mode']), ('conditions', r['condition']), ('splits', split)]:
            pieces[key].append(np.repeat(val, len(ids)))
    result = {k: np.concatenate(v) for k, v in pieces.items()}
    for name in ('raw', 'hoc', 'pi'):
        if not np.isfinite(result[name]).all():
            raise IntegrityError(f'Prepared {name} values are not finite.')
    if (result['raw'].ndim != 2 or result['raw'].shape[1] != length
            or result['hoc'].shape != (len(result['ids']), len(HOC_NAMES))
            or result['pi'].shape[0] != len(result['ids'])):
        raise IntegrityError('Prepared cache arrays have incompatible shapes.')
    if len(np.unique(result['ids'])) != len(result['ids']):
        raise IntegrityError('Duplicate source content/window IDs; inventory may contain copied recordings.')
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if set(result['groups'][result['splits'] == a]) & set(result['groups'][result['splits'] == b]):
            raise IntegrityError('Group leakage detected.')
    scaler_path = root/'cache'/cache_key/'scaler.json'
    if fit_missing_scaler:
        scaler = fit_scaler(result['hoc'], result['labels'], result['splits'], result['ids'])
        write_once(scaler_path, scaler)
    scaler = read_json(scaler_path)
    train_count = int(np.count_nonzero(result['splits'] == 'train'))
    if scaler['n_train'] != train_count or scaler['feature_names'] != HOC_NAMES:
        raise IntegrityError('Scaler metadata does not match the prepared training data.')
    result.update(scaler=scaler, feature_spec=spec, feature_key=cache_key,
                  data_plan_id=plan['plan_id'])
    return result
