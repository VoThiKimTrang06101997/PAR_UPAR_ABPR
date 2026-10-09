import ast
from pathlib import Path

def test_submission_imports_learned_ranker_only_if_packaged():
    src=(Path(__file__).resolve().parents[1]/'submission_run.py').read_text()
    ast.parse(src)
    assert 'ranker.npz' in src
    assert 'fuse_distances' in src
    assert 'torch.as_tensor(probs' in src
    assert 'attribute_prior.json' not in src


def test_builder_packages_selected_ranker():
    src=(Path(__file__).resolve().parents[1]/'build_codabench_submission.py').read_text()
    ast.parse(src)
    assert '--ranker-checkpoint' in src
    assert 'ranker.npz' in src
