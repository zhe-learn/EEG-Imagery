"""
Unified MI-EEG Training Script
===============================
Script for different models' training and evaluating.

Reads a YAML config (one per model) that carries both architecture and
training hyperparameters, including per-dataset / per-mode settings.

Usage:
    python main.py --config configs/eegnet.yaml
    python main.py --config configs/eegnet.yaml --dataset bcic_iv_2a
    python main.py --config configs/eegnet.yaml --dataset bcic_iv_2a --mode within --device auto

Without --dataset/--mode, every dataset (and every mode) listed in the YAML is run sequentially.

"""

import argparse
import importlib
import json
import logging
import os
import random
import re
import warnings
from datetime import datetime

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from load_data import deterministic_train_val_test_split, load_loso_data, load_subject_train_test

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified MI-EEG Training Experiment Settings")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to model YAML config file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="run only this dataset (default: all datasets in YAML)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["within", "cross"],
        help="run only this mode (default: all modes in YAML)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./datasets",
        help="root directory where dataset files are stored"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./_runs_/",
        help="base directory to save experiment results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="device to use: 'cpu', 'cuda', or 'auto' (auto selects cuda if available, else cpu)"
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="list of subjects to run (default: all subjects of the dataset)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 3, 5, 7, 9],
        help="random seeds (fixed for all experiments)"
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="random seed for data split (fixed for all experiments)"
    )

    return parser.parse_args()


#: model name -> (module, class name); all models share the same leading
#: dim parameters: in_channels, n_classes, n_samples
MODELS = {
    "atcnet": ("models.atcnet", "ATCNet"),
    "ctnet": ("models.ctnet", "CTNet"),
    "deepconvnet": ("models.deepconvnet", "DeepConvNet"),
    "eegnet": ("models.eegnet", "EEGNet"),
    "eegtcnet": ("models.eegtcnet", "EEGTCNet"),
    "factnet": ("models.factnet", "FACTNet"),
    "lmdanet": ("models.lmdanet", "LMDANet"),
    "model_mine": ("models.model_mine", "Net"),
    "sstdpn": ("models.sstdpn", "SSTDPN"),
}

DEFAULT_SUBJECTS = {
    "bcic_iv_2a": list(range(1, 10)),
    "bcic_iv_2b": list(range(1, 10)),
    "bcic_iii_4a": list(range(1, 6)),
    "wbcic_shu": list(range(1, 52)),
}

_DATASET_ORDER = ["bcic_iv_2a", "bcic_iv_2b", "bcic_iii_4a", "wbcic_shu"]


def _evaluate(model, X, y):
    """Evaluate model performance on given dataset"""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    acc = float(accuracy_score(y, preds))
    kappa = float(cohen_kappa_score(y, preds))
    return acc, kappa


def _save_history(save_path: str, history: dict):
    """Dump training history dict to JSON file, safely convert numpy scalars"""
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, default=float, indent=2)


def set_seed(seed):
    """Set Random Seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_experiment_dir(base_dir):
    """Make experiment directory"""
    os.makedirs(base_dir, exist_ok=True)
    existing_exps = []
    for entry in os.scandir(base_dir):
        if entry.is_dir():
            match = re.match(r'exp(\d+)', entry.name)
            if match:
                existing_exps.append(int(match.group(1)))
    new_exp_num = max(existing_exps) + 1 if existing_exps else 1
    exp_dir = os.path.join(base_dir, f"exp{new_exp_num}")
    os.makedirs(exp_dir)
    return exp_dir


def resolve_device(device_arg: str) -> str:
    """Parse device argument; 'auto' selects cuda if available, otherwise cpu"""
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def setup_logger(save_dir):
    """Set up training logger"""
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    log_file = f"{save_dir}/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    for handler in (logging.FileHandler(log_file), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def build_extra_loss(model_name, cfg_block, device):
    """Return callable (model, logits, labels) -> extra loss term, or None (CE only)"""
    if model_name == "model_mine":
        from models.model_mine import PrototypeDiscriminativeLoss
        pd = PrototypeDiscriminativeLoss(
            alpha=cfg_block.get("alpha", 0.01), beta=cfg_block.get("beta", 0.001)
        ).to(device)
        return lambda model, logits, labels: pd(model.features, model.proxy, labels)
    if model_name == "sstdpn":
        from models.sstdpn import PrototypeLoss, NormIncreaseLoss
        pl = PrototypeLoss().to(device)
        norm = NormIncreaseLoss().to(device)
        pw = cfg_block.get("pl_weight", 0.001)
        nw = cfg_block.get("norm_weight", 0.0001)
        return lambda model, logits, labels: (
                pw * pl(model.features, model.icp, labels) + nw * norm(model.icp)
        )
    return None


def train_one_run_two_stage(
        model,
        train_loader,
        combined_loader,
        X_val, y_val,
        X_test, y_test,
        criterion_ce,
        extra_loss,
        optimizer,
        stage1_epochs: int,
        stage2_epochs: int,
        device: str,
        subject: int,
        seed: int,
        run_dir: str,
        logger
) -> dict:
    """Two-stage training: Stage 1 on train set, Stage 2 on train+val combined set."""
    history = {
        "train_loss": [],
        "val_acc": [], "val_kappa": [],
        "test_acc": [], "test_kappa": [],
        "stage_boundary": stage1_epochs,
    }

    best_val_acc = 0.0
    best_val_kappa = 0.0
    best_val_epoch = 0

    best_stage1_model_path = os.path.join(run_dir, "best_val_stage1.pth")
    final_model_path = os.path.join(run_dir, "final_model.pth")

    # ==================== Stage 1 ====================
    logger.info(f"[S{subject} Seed={seed}] === Starting Stage 1 ({stage1_epochs} epochs) ===")

    pbar_stage1 = tqdm(
        range(stage1_epochs),
        desc=f"S{subject:02d}s{seed} S1",
        ncols=140,
        bar_format="{desc:12s}{percentage:4.0f}%|{bar:30}| {elapsed}<{remaining} {postfix}"
    )

    for epoch in pbar_stage1:
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion_ce(logits, labels)
            if extra_loss is not None:
                loss = loss + extra_loss(model, logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        history["train_loss"].append(float(avg_loss))

        val_acc, val_kappa = _evaluate(model, X_val, y_val)
        history["val_acc"].append(val_acc)
        history["val_kappa"].append(val_kappa)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_kappa = val_kappa
            best_val_epoch = epoch + 1
            torch.save(model.state_dict(), best_stage1_model_path)

        _save_history(os.path.join(run_dir, "history.json"), history)

        pbar_stage1.set_postfix_str(
            f"L={avg_loss:.3f} | "
            f"VA={val_acc:.3f} BK={best_val_acc:.3f}@{best_val_epoch}"
        )

    logger.info(
        f"[S{subject} Seed={seed}] Stage 1 Best Val: Acc={best_val_acc:.4f} "
        f"Kappa={best_val_kappa:.4f} @E{best_val_epoch}"
    )

    # ==================== Stage 2 ====================
    logger.info(f"[S{subject} Seed={seed}] Loading best Stage 1 model for Stage 2 training...")
    model.load_state_dict(
        torch.load(best_stage1_model_path, map_location=device, weights_only=True)
    )

    optimizer_stage2 = Adam(model.parameters(), lr=optimizer.param_groups[0]['lr'])
    logger.info(
        f"[S{subject} Seed={seed}] === Starting Stage 2 ({stage2_epochs} epochs) "
        f"with lr={optimizer_stage2.param_groups[0]['lr']} ==="
    )

    stage2_best_test_acc = 0.0
    stage2_best_test_kappa = 0.0
    stage2_best_test_epoch = 0

    final_test_acc = 0.0
    final_test_kappa = 0.0

    pbar_stage2 = tqdm(
        range(stage2_epochs),
        desc=f"S{subject:02d}s{seed} S2",
        ncols=140,
        bar_format="{desc:12s}{percentage:4.0f}%|{bar:30}| {elapsed}<{remaining} {postfix}"
    )

    for epoch in pbar_stage2:
        model.train()
        total_loss = 0.0
        for inputs, labels in combined_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_stage2.zero_grad()
            logits = model(inputs)
            loss = criterion_ce(logits, labels)
            if extra_loss is not None:
                loss = loss + extra_loss(model, logits, labels)
            loss.backward()
            optimizer_stage2.step()
            total_loss += loss.item() * inputs.size(0)

        avg_loss = total_loss / len(combined_loader.dataset)
        history["train_loss"].append(float(avg_loss))

        test_acc, test_kappa = _evaluate(model, X_test, y_test)
        history["test_acc"].append(test_acc)
        history["test_kappa"].append(test_kappa)

        if test_acc > stage2_best_test_acc:
            stage2_best_test_acc = test_acc
            stage2_best_test_kappa = test_kappa
            stage2_best_test_epoch = epoch + 1 + stage1_epochs

        if epoch == stage2_epochs - 1:
            final_test_acc, final_test_kappa = test_acc, test_kappa
            torch.save(model.state_dict(), final_model_path)

        _save_history(os.path.join(run_dir, "history.json"), history)

        pbar_stage2.set_postfix_str(
            f"L={avg_loss:.3f} | "
            f"TA={test_acc:.3f} BK={stage2_best_test_acc:.3f}@{stage2_best_test_epoch}"
        )

    logger.info(
        f"[S{subject} Seed={seed}] Stage 2 Best Test: Acc={stage2_best_test_acc:.4f} "
        f"Kappa={stage2_best_test_kappa:.4f} @E{stage2_best_test_epoch}"
    )
    logger.info(
        f"[S{subject} Seed={seed}] Stage 2 Final Test (last epoch): "
        f"Acc={final_test_acc:.4f}  Kappa={final_test_kappa:.4f}"
    )

    return {
        "stage1_best_val_acc": best_val_acc,
        "stage1_best_val_kappa": best_val_kappa,
        "stage1_best_val_epoch": best_val_epoch,
        "stage2_best_test_acc": stage2_best_test_acc,
        "stage2_best_test_kappa": stage2_best_test_kappa,
        "stage2_best_test_epoch": stage2_best_test_epoch,
        "final_test_acc": final_test_acc,
        "final_test_kappa": final_test_kappa,
        "history": history,
    }


def run_dataset_mode(
        model_name,
        arch_cfg,
        hp_cfg,
        extra_loss,
        seeds,
        split_seed,
        subjects,
        ds_key,
        mode,
        args,
        save_dir,
        logger
):
    """Train one model on one (dataset, mode) pair; write the two result files."""
    all_subjects = DEFAULT_SUBJECTS[ds_key]
    subjects = subjects if subjects is not None else all_subjects[:]
    val_subject = all_subjects[-1]

    stage1_results = {}
    final_results = {}

    for subject in subjects:
        logger.info(f"\n{'=' * 50}\nSUBJECT {subject}\n{'=' * 50}")

        if mode == "within":
            X_pool, y_pool, X_tst, y_tst = load_subject_train_test(args.data_root, ds_key, subject)
            X_tr, y_tr, X_v, y_v, _, _ = deterministic_train_val_test_split(
                X_pool, y_pool,
                train_ratio=0.8, val_ratio=0.2, test_ratio=0.0,
                seed=split_seed
            )
        else:
            X_tr, y_tr, X_v, y_v, X_tst, y_tst = load_loso_data(
                data_root=args.data_root,
                dataset_name=ds_key,
                test_subject=subject,
                all_subjects=all_subjects,
                val_subject=val_subject,
                split_seed=split_seed
            )

        n_ch, n_sp, n_classes = X_tr.shape[2], X_tr.shape[3], len(np.unique(y_tr))
        logger.info(
            f"Train Shape={X_tr.shape}  Val Shape={X_v.shape}  Test Shape={X_tst.shape}  Num Classes={n_classes}"
        )

        X_tr_t = torch.FloatTensor(X_tr).to(args.device)
        y_tr_t = torch.LongTensor(y_tr.flatten()).to(args.device)
        X_v_t = torch.FloatTensor(X_v).to(args.device)
        y_v_np = y_v.flatten()
        X_tst_t = torch.FloatTensor(X_tst).to(args.device)
        y_tst_np = y_tst.flatten()

        X_combined = np.concatenate([X_tr, X_v], axis=0)
        y_combined = np.concatenate([y_tr, y_v], axis=0)
        X_combined_t = torch.FloatTensor(X_combined).to(args.device)
        y_combined_t = torch.LongTensor(y_combined.flatten()).to(args.device)

        stage1_results[subject] = []
        final_results[subject] = []

        mod_path, cls_name = MODELS[model_name]
        ModelCls = getattr(importlib.import_module(mod_path), cls_name)

        for seed in seeds:
            set_seed(seed)
            model = ModelCls(in_channels=n_ch, n_classes=n_classes, n_samples=n_sp, **arch_cfg).to(args.device)

            train_dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=hp_cfg["batch_size"], shuffle=True
            )

            combined_dataset = torch.utils.data.TensorDataset(X_combined_t, y_combined_t)
            combined_loader = torch.utils.data.DataLoader(
                combined_dataset, batch_size=hp_cfg["batch_size"], shuffle=True
            )

            criterion_ce = nn.CrossEntropyLoss().to(args.device)
            optimizer = Adam(model.parameters(), lr=hp_cfg["lr"])

            run_dir = os.path.join(save_dir, f"sub{subject:02d}_seed{seed}")
            os.makedirs(run_dir, exist_ok=True)

            run_output = train_one_run_two_stage(
                model=model,
                train_loader=train_loader,
                combined_loader=combined_loader,
                X_val=X_v_t, y_val=y_v_np,
                X_test=X_tst_t, y_test=y_tst_np,
                criterion_ce=criterion_ce,
                extra_loss=extra_loss,
                optimizer=optimizer,
                stage1_epochs=hp_cfg["stage1_epochs"],
                stage2_epochs=hp_cfg["stage2_epochs"],
                device=args.device,
                subject=subject,
                seed=seed,
                run_dir=run_dir,
                logger=logger
            )

            stage1_results[subject].append(
                (run_output["stage1_best_val_acc"], run_output["stage1_best_val_kappa"])
            )
            final_results[subject].append(
                (run_output["final_test_acc"], run_output["final_test_kappa"])
            )

    # ==================== Save Two Result Files ====================
    stage1_file = os.path.join(save_dir, "stage1_best_val_results.txt")
    with open(stage1_file, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {ds_key}  Training Mode: {mode}\n")
        f.write(f"Two-stage: Stage1={hp_cfg['stage1_epochs']} epochs, Stage2={hp_cfg['stage2_epochs']} epochs\n")
        f.write(f"Random Seeds: {seeds}  Split Random Seed: {split_seed}\n")
        f.write("=" * 60 + "\n")
        f.write("STAGE 1 BEST VALIDATION RESULTS (Acc/Kappa per subject per seed)\n")
        f.write("=" * 60 + "\n\n")

        all_accs = []
        for subject in subjects:
            records = stage1_results[subject]
            acc_list = [r[0] for r in records]
            kappa_list = [r[1] for r in records]
            mean_acc = np.mean(acc_list)
            std_acc = np.std(acc_list, ddof=1) if len(acc_list) > 1 else 0.0
            mean_kappa = np.mean(kappa_list)
            std_kappa = np.std(kappa_list, ddof=1) if len(kappa_list) > 1 else 0.0
            all_accs.append(mean_acc)

            f.write(f"Subject {subject:2d}: Acc={mean_acc:.4f}±{std_acc:.4f}  Kappa={mean_kappa:.4f}±{std_kappa:.4f}\n")
            f.write(f"     Per seed Acc: {[f'{x:.4f}' for x in acc_list]}\n")
            f.write(f"     Per seed Kappa: {[f'{x:.4f}' for x in kappa_list]}\n\n")

        global_mean = np.mean(all_accs)
        global_std = np.std(all_accs, ddof=1) if len(all_accs) > 1 else 0.0
        f.write("=" * 60 + "\n")
        f.write(f"GLOBAL AVERAGE ACC: {global_mean:.4f}±{global_std:.4f}\n")

    stage2_final_file = os.path.join(save_dir, "stage2_final_test_results.txt")
    with open(stage2_final_file, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {ds_key}  Training Mode: {mode}\n")
        f.write(f"Two-stage: Stage1={hp_cfg['stage1_epochs']} epochs, Stage2={hp_cfg['stage2_epochs']} epochs\n")
        f.write(f"Random Seeds: {seeds}  Split Random Seed: {split_seed}\n")
        f.write("=" * 60 + "\n")
        f.write("STAGE 2 FINAL (LAST EPOCH) TEST RESULTS (Acc/Kappa per subject per seed)\n")
        f.write("=" * 60 + "\n\n")

        all_accs = []
        for subject in subjects:
            records = final_results[subject]
            acc_list = [r[0] for r in records]
            kappa_list = [r[1] for r in records]
            mean_acc = np.mean(acc_list)
            std_acc = np.std(acc_list, ddof=1) if len(acc_list) > 1 else 0.0
            mean_kappa = np.mean(kappa_list)
            std_kappa = np.std(kappa_list, ddof=1) if len(kappa_list) > 1 else 0.0
            all_accs.append(mean_acc)

            f.write(f"Subject {subject:2d}: Acc={mean_acc:.4f}±{std_acc:.4f}  Kappa={mean_kappa:.4f}±{std_kappa:.4f}\n")
            f.write(f"     Per seed Acc: {[f'{x:.4f}' for x in acc_list]}\n")
            f.write(f"     Per seed Kappa: {[f'{x:.4f}' for x in kappa_list]}\n\n")

        global_mean = np.mean(all_accs)
        global_std = np.std(all_accs, ddof=1) if len(all_accs) > 1 else 0.0
        f.write("=" * 60 + "\n")
        f.write(f"GLOBAL AVERAGE ACC: {global_mean:.4f}±{global_std:.4f}\n")

    logger.info(f"\n{'=' * 50}\nOVERALL EXPERIMENT SUMMARY\n{'=' * 50}")
    logger.info(f"Two result files saved to: {save_dir}")


def main():
    args = parse_args()

    seeds = args.seeds
    split_seed = args.split_seed
    args.device = resolve_device(args.device)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    model_name = cfg["model"]
    train_cfg = cfg.get("train", {}) or {}
    top_arch = cfg.get("arch", {}) or {}
    datasets_cfg = train_cfg.get("datasets") or {}

    if model_name not in MODELS:
        raise SystemExit(f"Unknown model '{model_name}' in {args.config}")

    # Which (dataset, mode) pairs to run
    ds_keys = [d for d in _DATASET_ORDER if d in datasets_cfg]
    if args.dataset:
        if args.dataset not in ds_keys:
            raise SystemExit(f"Dataset '{args.dataset}' not found in {args.config}")
        ds_keys = [args.dataset]

    pairs = []
    for ds in ds_keys:
        # only 'within'/'cross' are modes; other keys (arch/seeds/...) are meta
        modes = [m for m in ("within", "cross") if m in datasets_cfg[ds]]
        if args.mode:
            if args.mode not in modes:
                raise SystemExit(f"Mode '{args.mode}' not configured for {ds} in {args.config}")
            modes = [args.mode]
        for mode in modes:
            pairs.append((ds, mode))

    # subject selection: CLI --subjects > train.subjects in YAML > all subjects
    yaml_subjects = train_cfg.get("subjects")
    subjects = args.subjects if args.subjects is not None else yaml_subjects

    for ds_key, mode in pairs:
        ds_cfg = datasets_cfg[ds_key]
        # per-dataset settings, falling back to top-level defaults when absent
        arch_cfg = ds_cfg.get("arch") or top_arch
        hp_cfg = ds_cfg[mode]
        extra_loss = build_extra_loss(model_name, ds_cfg, args.device)

        save_dir = create_experiment_dir(os.path.join(args.base_dir, ds_key, mode))
        logger = setup_logger(save_dir)
        logger.info(
            f"Model={model_name} | Dataset={ds_key} | Mode={mode} | Config={args.config} | "
            f"Device={args.device} | Seeds={seeds} | SplitSeed={split_seed}"
        )
        logger.info(
            f"Two-stage training: Stage1={hp_cfg['stage1_epochs']} epochs, Stage2={hp_cfg['stage2_epochs']} epochs | "
            f"Batch={hp_cfg['batch_size']} | Lr={hp_cfg['lr']}"
        )
        logger.info(f"Experiment Output Directory: {save_dir}")

        run_dataset_mode(
            model_name, arch_cfg, hp_cfg, extra_loss, seeds, split_seed, subjects, ds_key, mode, args, save_dir, logger
        )


if __name__ == "__main__":
    main()
