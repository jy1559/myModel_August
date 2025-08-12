import importlib
from typing import Dict

_ALIAS = {
    "mymodel": "myModel",
    "narm":     "NARM",
    "sr-gnn":    "SR-GNN",
    "h-rnn":    "H-RNN",
    "gag":    "GAG",
    "hg-gnn":    "HG-GNN",
    "hiertcn":    "HierTCN",
    "tisasrec":  "TiSASRec",
}

def build_model(name: str, n_items: int, cfg: Dict):
    """
    Parameters
    ----------
    name     : one of {'baseline','narm','srgnn', ...}
    n_items  : vocabulary size
    cfg      : hyper-parameter dict (CLI args 그대로 전달)
    """
    key = name.lower()
    if key not in _ALIAS:
        raise ValueError(f"[models] unknown model '{name}'. "
                         f"available: {list(_ALIAS)}")

    module_path = f"models.{_ALIAS[key]}.model"
    mod = importlib.import_module(module_path)

    if not hasattr(mod, "SeqRecModel"):
        raise ImportError(f"{module_path} must expose `SeqRecModel`.")
    return mod.SeqRecModel(n_items, cfg)