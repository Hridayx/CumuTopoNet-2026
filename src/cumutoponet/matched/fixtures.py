"""Small invented waveforms for SOFTWARE TESTING ONLY, never research evidence."""
from pathlib import Path
import numpy as np
import yaml


def create_fixture(directory):
    directory = Path(directory).resolve()
    root = directory/'dataset'
    rng = np.random.default_rng(123)
    for label, drone in enumerate(['AIR', 'INS', 'MIN', 'MP1', 'MP2', 'PHA', 'DIS']):
        for condition in ['CLEAN', 'BLUE']:
            for mode in ['FY', 'HO', 'ON']:
                path = root/'fixture'/condition/f'{drone}_{mode}'/'capture.dat'
                if path.exists():
                    raise FileExistsError('Use a fresh fixture output directory.')
                path.parent.mkdir(parents=True, exist_ok=True)
                t = np.arange(8192)
                x = np.exp(2j*np.pi*(label+1)*t/32) + .2*(rng.normal(size=len(t))+1j*rng.normal(size=len(t)))
                x.astype('<c8').tofile(path)
    config = {'fixture': True, 'dataset': {'path': str(root), 'windows_per_recording': 2},
              'workspace': str(directory/'work'), 'runtime': {'cpus': 2, 'archive_readers': 1, 'feature_workers': 1},
              'training': {'batch_size': 14, 'max_epochs': 2, 'min_epochs': 2, 'amp': False,
                           'checkpoint_steps': 2}, 'budget': {'hours': 72, 'gpu_workers': 2}}
    config_path = directory/'fixture.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path
