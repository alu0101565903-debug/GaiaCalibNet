import importlib

MODULES_TO_IMPORT = [
    "src.processing.crossmatch_wd",
    "src.processing.download_calspec",
    "src.processing.prepare_dataset",
    "src.processing.extract_features",
    "src.models.neural_net",
    "src.models.baselines",
    "src.train.mains.train_baselines",
    "src.train.mains.train_neural_net",
    "src.experiments.evaluate_catalog",
]


def test_imports():
    for module_name in MODULES_TO_IMPORT:
        importlib.import_module(module_name)
