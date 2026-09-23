from cumutoponet.matched.common import load_config as load_matched
from cumutoponet.full.common import load_config as load_full


def test_config_paths_are_relative_to_yaml_file(tmp_path):
    directory = tmp_path / "configs"
    directory.mkdir()
    config = directory / "local.yaml"
    config.write_text("dataset:\n  path: ../data/input.zip\nworkspace: ../work/study\n")
    for load in (load_matched, load_full):
        loaded = load(config)
        assert loaded["dataset"]["path"] == str((tmp_path / "data/input.zip").resolve())
        assert loaded["workspace"] == str((tmp_path / "work/study").resolve())
