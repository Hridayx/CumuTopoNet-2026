import time
import pytest
from cumutoponet import SUPCON_WEIGHT
from cumutoponet.matched.common import write_json, read_json, IntegrityError
from cumutoponet.matched.protocol import budget_plan, start_pilots, advance
from cumutoponet.matched.benchmark import PRINCIPAL
from cumutoponet.matched.data import inventory, prepare_dataset
from cumutoponet.matched.queue import JobQueue, worker_identity
from test_data import fixture_config


def measured_fixture(cfg, seconds=1.5):
    from pathlib import Path
    root = Path(cfg['workspace'])
    inv = inventory(cfg)
    records = [{key: record[key] for key in ('id', 'path', 'bytes', 'samples', 'source_metadata')}
               for record in inv['records']]
    write_json(root/'benchmarks'/'cpu.json', {'inventory_source': inv['source'],
        'inventory_records': records, 'recordings': len(inv['records']), 'total_bytes': inv['total_bytes'],
        'read_bytes_per_second': 1e8, 'feature_seconds_per_window': {'128': .001, '1024': .001}})
    write_json(root/'benchmarks'/'gpu.json', {'timings': {
        f'{name}:{length}': {'train_step_seconds': seconds, 'validation_step_seconds': seconds/4,
                            'batch_size': 256} for name in PRINCIPAL for length in [128, 1024]}})


def test_budget_reduces_sampling_before_production_and_cannot_change_afterwards(tmp_path):
    cfg = fixture_config(tmp_path)
    cfg['dataset']['windows_per_recording'] = 1024
    cfg['fixture'] = False
    cfg['runtime'].update(cpus=52, archive_readers=4, feature_workers=48)
    measured_fixture(cfg)
    decision = budget_plan(cfg)
    assert decision['windows_per_recording'] == 512
    assert decision['feasible']
    assert budget_plan(cfg) == decision
    measured_fixture(cfg, seconds=.001)
    with pytest.raises(IntegrityError, match='frozen'):
        budget_plan(cfg)


def test_protocol_has_33_core_fits_after_learning_rate_pilots(tmp_path):
    cfg = fixture_config(tmp_path)
    measured_fixture(cfg, .001)
    budget_plan(cfg)
    prepare_dataset(cfg)
    q = JobQueue(cfg['workspace'])
    assert len(start_pilots(cfg)['jobs']) == 8
    assert advance(cfg)['stage'] == 'pilot-lr-running'
    from pathlib import Path
    root = Path(cfg['workspace'])
    def complete_pending():
        while (claim := q.claim(worker_identity('fixture'))) is not None:
            spec = claim['spec']
            write_json(root/'runs'/claim['id']/'result.json', {
                'spec': spec, 'best_validation_macro_f1': .5})
            q.finish(claim, 'completed')
    complete_pending()
    assert advance(cfg)['principal_fits'] == len(PRINCIPAL) * len(cfg['training']['seeds'])
    protocol = read_json(root/'protocol.json')
    assert len(protocol['core_specs']) == len(PRINCIPAL) * len(cfg['training']['seeds'])
    assert {s['seed'] for s in protocol['core_specs']} == {17, 42, 91}
    assert protocol['supcon_weight'] == SUPCON_WEIGHT
    assert set(protocol['selected_learning_rates'].values()) == {.0003}
    assert all(s['feature_key'] == protocol['core_specs'][0]['feature_key'] for s in protocol['core_specs'])
    assert {s['supcon_weight'] for s in protocol['core_specs']} <= {0., SUPCON_WEIGHT}
