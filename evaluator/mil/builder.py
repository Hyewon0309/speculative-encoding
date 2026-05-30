"""Build MIL network from config (arch name + feature dim, n_class, etc.)."""

import torch

from . import clam
from . import dftd
from . import dsmil
from . import ilra
from . import mean_max
from . import rrt
from . import transmil
from . import transformer
from . import wikg


def make_conf(
    arch: str,
    feature_dim: int,
    n_class: int = 2,
    train_epoch: int = 50,
    lr: float = 0.0001,
    wd: float = 1e-5,
    n_token: int = 1,
    w_loss: float = 1.0,
    warmup_epoch: int = 0,
    min_lr: float = 0.0,
    eval_interval: int = 5,
):
    """Build a config object for MIL (Struct-like)."""
    D_feat = feature_dim
    D_inner = 128 if D_feat <= 384 else (256 if D_feat <= 512 else 384)
    return type("Conf", (), {
        "arch": arch,
        "D_feat": D_feat,
        "D_inner": D_inner,
        "feat_d": D_feat,
        "n_class": n_class,
        "train_epoch": train_epoch,
        "lr": lr,
        "wd": wd,
        "min_lr": min_lr,
        "warmup_epoch": warmup_epoch,
        "n_token": n_token,
        "w_loss": w_loss,
        "eval_interval": eval_interval,
        "B": 1,
        "n_worker": 0,
        "pin_memory": False,
    })()


def build_net(conf, device: torch.device) -> torch.nn.Module:
    """Instantiate MIL architecture from conf."""
    if conf.arch == "transmil":
        net = transmil.TransMIL(conf)
    elif conf.arch == "mha":
        net = transformer.MHA(conf)
    elif conf.arch == "clam_sb":
        net = clam.CLAM_SB(conf).to(device)
    elif conf.arch == "clam_mb":
        net = clam.CLAM_MB(conf).to(device)
    elif conf.arch == "dsmil":
        i_classifier = dsmil.FCLayer(conf.D_feat, conf.n_class)
        b_classifier = dsmil.BClassifier(conf, nonlinear=False)
        net = dsmil.MILNet(i_classifier, b_classifier)
    elif conf.arch == "bmil_spvis":
        raise NotImplementedError(
            "bmil_spvis is not vendored in mil; use PathGen or add architecture/bmil.py"
        )
    elif conf.arch == "abmil":
        net = transformer.ABMIL(conf)
    elif conf.arch == "meanmil":
        net = mean_max.MeanMIL(conf).to(device)
    elif conf.arch == "maxmil":
        net = mean_max.MaxMIL(conf).to(device)
    elif conf.arch == "ilra":
        net = ilra.ILRA(feat_dim=conf.D_feat, n_classes=conf.n_class, ln=True)
    elif conf.arch == "dftd":
        net = dftd.DFTD(conf)
    elif conf.arch == "rrt":
        net = rrt.RRT(conf)
    elif conf.arch == "wikg":
        net = wikg.WiKG(conf)
    else:
        raise ValueError(f"Unknown arch: {conf.arch}")
    net.to(device)
    return net
