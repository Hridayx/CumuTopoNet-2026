from cumutoponet import SUPCON_WEIGHT
from cumutoponet.cli import parser
from cumutoponet.full.common import DEFAULTS
from cumutoponet.full.extended import extended_specs
from cumutoponet.full.protocol import experiment_specs


def test_public_cli_and_suite_names():
    parsed = parser().parse_args([
        "full-1024", "plan", "-c", "local.yaml", "--suite", "standard-1024",
    ])
    assert parsed.suite == "full-1024"
    assert parsed.experiment_suite == "standard-1024"
    report = parser().parse_args([
        "matched-128", "report", "-c", "local.yaml", "--output-dir", "reports",
    ])
    assert report.output_dir == "reports"


def test_supcon_weight_is_fixed_for_generated_specs():
    specs = experiment_specs(DEFAULTS) + extended_specs(DEFAULTS)
    assert {spec["supcon_weight"] for spec in specs} <= {0., SUPCON_WEIGHT}
    assert all(spec["supcon_weight"] == SUPCON_WEIGHT
               for spec in specs if spec["model"] not in {"full_ce", "raw_iq", "lstm", "spectrogram_cnn"})
