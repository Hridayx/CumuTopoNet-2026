from pathlib import Path
import zipfile
import numpy as np
import pytest
from cumutoponet.matched.common import DEFAULTS, merge, IntegrityError
from cumutoponet.matched.data import inventory, sample_offsets, read_selected, make_partitions, prepare_dataset, load_arrays


def make_dataset(root, repetitions=1):
    rng = np.random.default_rng(8)
    for drone in ['AIR', 'INS', 'MIN', 'MP1', 'MP2', 'PHA', 'DIS']:
        for condition in ['CLEAN', 'BLUE']:
            for mode in ['FY', 'HO', 'ON']:
                for rep in range(repetitions):
                    p = root / 'nested' / condition / f'{drone}_{mode}' / f'capture_{rep}.dat'
                    p.parent.mkdir(parents=True, exist_ok=True)
                    rng.normal(size=(8192, 2)).astype('<f4').tofile(p)


def fixture_config(tmp_path):
    root = tmp_path / 'dataset'
    make_dataset(root)
    return merge(DEFAULTS, {'dataset': {'path': str(root), 'windows_per_recording': 2},
                           'workspace': str(tmp_path/'work'), 'fixture': True,
                           'runtime': {'cpus': 2, 'archive_readers': 1, 'feature_workers': 1}})


def test_zip_parity_and_family_separation(tmp_path):
    root = tmp_path / 'data'
    make_dataset(root, repetitions=2)
    archive = tmp_path / 'data.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*.dat')):
            z.write(p, p.relative_to(root).as_posix())
    cfg = merge(DEFAULTS, {'dataset': {'path': str(root)}})
    a = inventory(cfg)
    cfg['dataset']['path'] = str(archive)
    b = inventory(cfg)
    assert len(a['records']) == len(b['records']) == 84
    offsets = sample_offsets(8192, 4, 1024, seed=17)
    assert np.all(np.diff(offsets) >= 1024)
    x = read_selected(str(root), a['records'][0], offsets, 1024)
    y = read_selected(str(archive), b['records'][0], offsets, 1024)
    np.testing.assert_array_equal(x, y)
    split = make_partitions(a['records'])
    for family in {r['family'] for r in a['records']}:
        assert len({split[r['id']] for r in a['records'] if r['family'] == family}) == 1
    for partition in range(5):
        assert len({r['label'] for r in a['records'] if split[r['id']] == partition}) == 7


def test_prepared_data_resumes_and_rejects_corrupt_shards(tmp_path):
    cfg = fixture_config(tmp_path)
    prepare_dataset(cfg)
    first = load_arrays(cfg, length=128)
    prepare_dataset(cfg)
    second = load_arrays(cfg, length=128)
    np.testing.assert_array_equal(first['ids'], second['ids'])
    assert len(first['ids']) == 84
    assert len(set(first['ids'])) == 84
    assert first['scaler']['n_train'] == int((first['splits'] == 'train').sum())
    # An unreadable completed cache must be diagnosed, not regenerated silently.
    p = next((Path(cfg['workspace'])/'cache'/'raw').glob('*/parent.npy'))
    with p.open('r+b') as f:
        f.truncate(p.stat().st_size - 16)
    with pytest.raises(IntegrityError, match='Unreadable'):
        load_arrays(cfg, length=128)


def test_scaler_ignores_held_out_values():
    from cumutoponet.matched.features import fit_scaler
    x = np.array([[1]*6, [3]*6, [1e10]*6, [-1e10]*6], dtype=float)
    result = fit_scaler(x, np.arange(4), np.array(['train', 'train', 'validation', 'test']), np.array(['a','b','c','d']))
    np.testing.assert_allclose(result['mean'], [2]*6)
    np.testing.assert_allclose(result['scale'], [1]*6)


def test_cache_request_mismatch_is_detected_directly(tmp_path):
    from cumutoponet.matched.common import read_json, write_json
    cfg = fixture_config(tmp_path)
    prepare_dataset(cfg)
    marker = next((Path(cfg['workspace'])/'cache'/'raw').glob('*/meta.json'))
    value = read_json(marker)
    value['request']['count'] += 1
    write_json(marker, value)
    with pytest.raises(IntegrityError, match='does not match|differs'):
        load_arrays(cfg, length=128)


def test_stream_retains_a_window_crossing_the_read_chunk_boundary(tmp_path):
    count = (1 << 20) + 2048
    x = (np.arange(count, dtype=np.float32) + 1j*np.arange(count, dtype=np.float32)[::-1]).astype('<c8')
    x.tofile(tmp_path/'record.dat')
    offsets = np.array([10, (1 << 20)-76])
    actual = read_selected(str(tmp_path), {'path': 'record.dat', 'samples': count, 'bytes': count*8}, offsets)
    np.testing.assert_array_equal(actual[0], x[10:1034])
    np.testing.assert_array_equal(actual[1], x[offsets[1]:offsets[1]+1024])


def test_session_metadata_can_merge_but_cannot_split_families(tmp_path):
    import csv
    root = tmp_path/'data'
    make_dataset(root, repetitions=2)
    cfg = merge(DEFAULTS, {'dataset': {'path': str(root)}})
    first = inventory(cfg)
    rows = []
    merged = [r for r in first['records'] if r['label_name'] == 'Air2S' and r['mode'] == 'Flying']
    for i, r in enumerate(first['records']):
        # Different IDs for repeats do not split the family; one shared ID coarsens two families.
        session = 'shared-acquisition' if r in merged and r['path'].endswith('capture_0.dat') else f'file-{i}'
        rows.append({'relative_path': r['path'], 'session_id': session})
    path = tmp_path/'sessions.csv'
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['relative_path', 'session_id'])
        w.writeheader()
        w.writerows(rows)
    cfg['dataset']['session_metadata'] = str(path)
    second = inventory(cfg)
    assert len({r['group'] for r in second['records'] if r['id'] in {m['id'] for m in merged}}) == 1
