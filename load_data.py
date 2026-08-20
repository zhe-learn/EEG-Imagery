"""
MI-EEG Dataset Loading Utilities
=================================
Unified data loading with deterministic stratified train/val/test splits and Leave-One-Subject-Out (LOSO) support.

All splits use pure NumPy — results are byte-identical across environments for a given random seed.

Supported datasets:
    bcic_iv_2a   - BCI Competition IV Dataset 2a   (22  ch,   4-class,  9  subj)
    bcic_iv_2b   - BCI Competition IV Dataset 2b   (3   ch,   2-class,  9  subj)
    bcic_iii_4a  - BCI Competition III Dataset 4a  (118 ch,   2-class,  5  subj)
    wbcic_shu    - WBCIC-SHU Dataset               (58  ch,   2-class,  52 subj)

"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat


# ==============================================================================
# Internal: single-subject loaders
# ==============================================================================

def _load_bcic_iv_2a(data_root: str, subject: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ALL trials for one BCIC IV 2a subject.

    Returns:
        X  [N, 1, 22, 1000]
        y  [N,]   labels 0-3
    """
    if not (1 <= subject <= 9):
        raise ValueError(f"BCIC IV 2a subject must be 1-9, got {subject}")

    data_path = os.path.join(data_root, 'BCIC-IV-2a')
    fs, n_ch = 250, 22
    win_len = 7 * fs
    t1, t2 = 2 * fs, 6 * fs

    X_list, y_list = [], []
    for suffix in ('T', 'E'):
        fpath = os.path.join(data_path, f"A0{subject}{suffix}.mat")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"BCIC IV 2a file not found: {fpath}")
        mat = loadmat(fpath)['data']
        for i in range(mat.size):
            trial_data = mat[0, i][0, 0][0]
            trial_starts = mat[0, i][0, 0][1]
            trial_labels = mat[0, i][0, 0][2]  # 1-4
            for t in range(trial_starts.size):
                start = int(trial_starts[t].item())
                eeg = np.transpose(trial_data[start:start + win_len, :n_ch])
                X_list.append(eeg[:, t1:t2])
                y_list.append(int(trial_labels[t].item()))

    X = np.stack(X_list, axis=0)  # [N, 22, 1000]
    X = np.expand_dims(X, axis=1)  # [N, 1, 22, 1000]
    y = np.array(y_list, dtype=np.int64) - 1  # 0-based
    return X, y


def _load_bcic_iv_2a_train_test(
        data_root: str, subject: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load BCIC IV 2a with the original competition split.

    Returns:
        X_train_pool: Training data [n_train, 1, 22, 1000]
        y_train_pool: Training labels [n_train,]
        X_test: Test data [n_test, 1, 22, 1000]
        y_test: Test labels [n_test,]
    """
    if not (1 <= subject <= 9):
        raise ValueError(f"BCIC IV 2a subject must be 1-9, got {subject}")

    data_path = os.path.join(data_root, 'BCIC-IV-2a')
    fs, n_ch = 250, 22
    win_len = 7 * fs
    t1, t2 = 2 * fs, 6 * fs

    def _load_one(suffix):
        xs, ys = [], []
        fpath = os.path.join(data_path, f"A0{subject}{suffix}.mat")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"BCIC IV 2a file not found: {fpath}")
        mat = loadmat(fpath)['data']
        for i in range(mat.size):
            td = mat[0, i][0, 0][0]
            ts = mat[0, i][0, 0][1]
            tl = mat[0, i][0, 0][2]
            for t in range(ts.size):
                start = int(ts[t].item())
                eeg = np.transpose(td[start:start + win_len, :n_ch])
                xs.append(eeg[:, t1:t2])
                ys.append(int(tl[t].item()))
        X = np.stack(xs, axis=0)
        X = np.expand_dims(X, axis=1)
        y = np.array(ys, dtype=np.int64) - 1
        return X, y

    X_train_pool, y_train_pool = _load_one('T')
    X_test, y_test = _load_one('E')
    return X_train_pool, y_train_pool, X_test, y_test


def _load_bcic_iv_2b(data_root: str, subject: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ALL trials for one BCIC IV 2b subject.

    Returns:
        X  [N, 1, 3, 1000]
        y  [N,]   labels 0-1
    """
    if not (1 <= subject <= 9):
        raise ValueError(f"BCIC IV 2b subject must be 1-9, got {subject}")

    data_path = os.path.join(data_root, 'BCIC-IV-2b')
    fs, n_ch = 250, 3
    win_len = 7 * fs
    t1, t2 = 3 * fs, 7 * fs

    X_list, y_list = [], []
    for suffix in ('T', 'E'):
        fpath = os.path.join(data_path, f"B0{subject}{suffix}.mat")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"BCIC IV 2b file not found: {fpath}")
        mat = loadmat(fpath)['data']
        for i in range(mat.size):
            trial_data = mat[0, i][0, 0][0]
            trial_starts = mat[0, i][0, 0][1]
            trial_labels = mat[0, i][0, 0][2]  # 1-2
            for t in range(trial_starts.size):
                start = int(trial_starts[t].item())
                eeg = np.transpose(trial_data[start:start + win_len, :n_ch])
                X_list.append(eeg[:, t1:t2])
                y_list.append(int(trial_labels[t].item()))

    X = np.stack(X_list, axis=0)
    X = np.expand_dims(X, axis=1)  # [N, 1, 3, 1000]
    y = np.array(y_list, dtype=np.int64) - 1  # 0-based
    return X, y


def _load_bcic_iv_2b_train_test(
        data_root: str, subject: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load BCIC IV 2b with the original competition split.

    Returns:
        X_train_pool: Training data [n_train, 1, 3, 1000]
        y_train_pool: Training labels [n_train,]
        X_test: Test data [n_test, 1, 3, 1000]
        y_test: Test labels [n_test,]
    """
    if not (1 <= subject <= 9):
        raise ValueError(f"BCIC IV 2b subject must be 1-9, got {subject}")

    data_path = os.path.join(data_root, 'BCIC-IV-2b')
    fs, n_ch = 250, 3
    win_len = 7 * fs
    t1, t2 = 3 * fs, 7 * fs

    def _load_one(suffix):
        xs, ys = [], []
        fpath = os.path.join(data_path, f"B0{subject}{suffix}.mat")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"BCIC IV 2b file not found: {fpath}")
        mat = loadmat(fpath)['data']
        for i in range(mat.size):
            td = mat[0, i][0, 0][0]
            ts = mat[0, i][0, 0][1]
            tl = mat[0, i][0, 0][2]
            for t in range(ts.size):
                start = int(ts[t].item())
                eeg = np.transpose(td[start:start + win_len, :n_ch])
                xs.append(eeg[:, t1:t2])
                ys.append(int(tl[t].item()))
        X = np.stack(xs, axis=0)
        X = np.expand_dims(X, axis=1)
        y = np.array(ys, dtype=np.int64) - 1
        return X, y

    X_train_pool, y_train_pool = _load_one('T')
    X_test, y_test = _load_one('E')
    return X_train_pool, y_train_pool, X_test, y_test


def _load_bcic_iii_4a(data_root: str, subject: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ALL trials for one BCIC III 4a subject (Only channels C3, Cz, C4 are selected).

    Returns:
        X  [N, 1, 3, 350]
        y  [N,]   labels 0-1
    """
    sub_suffix = {1: "aa", 2: "al", 3: "av", 4: "aw", 5: "ay"}
    if subject not in sub_suffix:
        raise ValueError(f"BCIC III 4a subject must be 1-5, got {subject}")

    suff = sub_suffix[subject]
    data_path = os.path.join(data_root, 'BCIC-III-4a')

    eeg_path = os.path.join(
        data_path, f"data_set_IVa_{suff}_mat", "100Hz",
        f"data_set_IVa_{suff}.mat"
    )
    label_path = os.path.join(data_path, f"true_labels_{suff}.mat")

    if not os.path.exists(eeg_path):
        raise FileNotFoundError(f"BCIC III 4a data file not found: {eeg_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"BCIC III 4a label file not found: {label_path}")

    mat_eeg = loadmat(eeg_path)
    mat_labels = loadmat(label_path)

    cnt = mat_eeg["cnt"].T  # [n_channels, n_time_points]
    markers = mat_eeg["mrk"][0][0][0]  # trial start indices
    y = mat_labels["true_y"]  # labels (1-2)

    # Get channel names and select C3, Cz, C4
    ch_names = [ch[0] for ch in mat_eeg["nfo"]["clab"][0][0][0]]
    selected_channels = ["C3", "Cz", "C4"]

    # Check if all selected channels exist
    missing_channels = [ch for ch in selected_channels if ch not in ch_names]
    if missing_channels:
        raise ValueError(
            f"Channels not found: {missing_channels}. "
            f"Available channels (first 10): {ch_names[:10]}"
        )

    # Select channels
    channel_indices = [ch_names.index(ch) for ch in selected_channels]
    cnt = cnt[channel_indices, :]  # [3, n_time_points]

    sfreq = mat_eeg["nfo"]["fs"][0][0][0][0]  # 100 Hz
    trial_len = int(sfreq * 3.5)  # 350 samples
    n_trials = y.shape[-1]
    n_ch = cnt.shape[0]  # 3 channels

    X = np.zeros((n_trials, 1, n_ch, trial_len), dtype=np.float64)
    for i, marker in enumerate(markers[0]):
        m = int(marker)
        X[i, 0, :, :] = cnt[:, m:m + trial_len]

    y = (y.flatten() - 1).astype(np.int64)  # 0-based
    return X, y


def _load_bcic_iii_4a_train_test(
        data_root: str, subject: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load BCIC III 4a with the original competition split (Only channels C3, Cz, C4 are selected).

    Returns:
        X_train_pool: Training data [n_train, 1, 3, 350]
        y_train_pool: Training labels [n_train,]
        X_test: Test data [n_test, 1, 3, 350]
        y_test: Test labels [n_test,]
    """
    sub_suffix = {1: "aa", 2: "al", 3: "av", 4: "aw", 5: "ay"}
    if subject not in sub_suffix:
        raise ValueError(f"BCIC III 4a subject must be 1-5, got {subject}")

    suff = sub_suffix[subject]
    data_path = os.path.join(data_root, 'BCIC-III-4a')

    eeg_path = os.path.join(
        data_path, f"data_set_IVa_{suff}_mat", "100Hz",
        f"data_set_IVa_{suff}.mat"
    )
    label_path = os.path.join(data_path, f"true_labels_{suff}.mat")

    if not os.path.exists(eeg_path):
        raise FileNotFoundError(f"BCIC III 4a data file not found: {eeg_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"BCIC III 4a label file not found: {label_path}")

    mat_eeg = loadmat(eeg_path)
    mat_labels = loadmat(label_path)

    cnt = mat_eeg["cnt"].T  # [n_channels, n_time_points]
    markers = mat_eeg["mrk"][0][0][0]  # trial start indices
    y = mat_labels["true_y"]  # labels (1-2)
    test_idx = mat_labels["test_idx"][0, 0]  # Split index for train/test

    # Get channel names and select C3, Cz, C4
    ch_names = [ch[0] for ch in mat_eeg["nfo"]["clab"][0][0][0]]
    selected_channels = ["C3", "Cz", "C4"]

    # Check if all selected channels exist
    missing_channels = [ch for ch in selected_channels if ch not in ch_names]
    if missing_channels:
        raise ValueError(
            f"Channels not found: {missing_channels}. "
            f"Available channels (first 10): {ch_names[:10]}"
        )

    # Select channels
    channel_indices = [ch_names.index(ch) for ch in selected_channels]
    cnt = cnt[channel_indices, :]  # [3, n_time_points]

    sfreq = mat_eeg["nfo"]["fs"][0][0][0][0]  # 100 Hz
    trial_len = int(sfreq * 3.5)  # 350 samples
    n_trials = y.shape[-1]
    n_ch = cnt.shape[0]  # 3 channels

    # Extract all trials
    X = np.zeros((n_trials, 1, n_ch, trial_len), dtype=np.float64)
    for i, marker in enumerate(markers[0]):
        m = int(marker)
        X[i, 0, :, :] = cnt[:, m:m + trial_len]

    # Convert labels to 0-based
    y = (y.flatten() - 1).astype(np.int64)

    # Split into train and test sets based on test_idx
    # test_idx is 1-based, so convert to 0-based indexing
    split_idx = test_idx - 1

    X_train_pool = X[:split_idx]
    y_train_pool = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    return X_train_pool, y_train_pool, X_test, y_test


def _load_wbcic_shu(data_root: str, subject: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ALL trials for one WBCIC SHU subject.

    Returns:
        X  [N, 1, 58, 1000]
        y  [N,]   labels 0-1
    """
    if not (1 <= subject <= 52):
        raise ValueError(f"WBCIC-SHU subject must be 1-52, got {subject}")

    data_path = os.path.join(data_root, 'WBCIC-SHU')
    subject_str = f"{subject:03d}"  # 1 → 001

    # Define all sessions (01-03)
    all_sessions = ['01', '02', '03']

    X_list, y_list = [], []

    for ses in all_sessions:
        file_path = os.path.join(
            data_path,
            f"Sub-{subject_str}",
            f"dataset1_processeddata_Sub-{subject_str}_sess-{ses}_task-MI_eeg.mat"
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"WBCIC-SHU file not found: {file_path}")

        mat_data = loadmat(file_path)

        # Extract data: [n_channels, n_time_points, n_trials]
        data = np.array(mat_data['data'])

        # Extract labels: [1, n_trials] -> squeeze to [n_trials,]
        labels = np.array(mat_data['labels']).squeeze()

        # Transpose from [n_channels, n_time_points, n_trials] to [n_trials, n_channels, n_time_points]
        data = np.transpose(data, (2, 0, 1))  # [n_trials, n_channels, n_time_points]

        X_list.append(data)
        y_list.append(labels)

    # Concatenate all sessions
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # Add channel dimension: [N, 1, n_channels, n_time_points]
    X = np.expand_dims(X, axis=1)

    # Convert labels to 0-based if they start from 1
    if np.min(y) == 1:
        y = y - 1

    y = y.astype(np.int64)
    return X, y


def _load_wbcic_shu_train_test(
        data_root: str, subject: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load WBCIC-SHU with fixed session split:
        Session 01 → training pool
        Sessions 02, 03 → test set

    Returns:
        X_train_pool: Training data [n_train, 1, 58, 1000]
        y_train_pool: Training labels [n_train,]
        X_test: Test data [n_test, 1, 58, 1000]
        y_test: Test labels [n_test,]
    """
    if not (1 <= subject <= 52):
        raise ValueError(f"WBCIC-SHU subject must be 1-52, got {subject}")

    data_path = os.path.join(data_root, 'WBCIC-SHU')
    subject_str = f"{subject:03d}"

    def _load_sessions(session_list):
        xs, ys = [], []
        for ses in session_list:
            file_path = os.path.join(
                data_path,
                f"Sub-{subject_str}",
                f"dataset1_processeddata_Sub-{subject_str}_sess-{ses}_task-MI_eeg.mat"
            )
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"WBCIC-SHU file not found: {file_path}")

            mat_data = loadmat(file_path)
            data = np.array(mat_data['data'])  # [n_channels, n_time_points, n_trials]
            labels = np.array(mat_data['labels']).squeeze()  # [n_trials,]

            # Transpose to [n_trials, n_channels, n_time_points]
            data = np.transpose(data, (2, 0, 1))

            xs.append(data)
            ys.append(labels)

        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        X = np.expand_dims(X, axis=1)  # [N, 1, n_channels, n_time_points]
        if np.min(y) == 1:
            y = y - 1
        return X, y.astype(np.int64)

    # Session 01 → training pool, Sessions 02-03 → test set
    X_train_pool, y_train_pool = _load_sessions(['01'])
    X_test, y_test = _load_sessions(['02', '03'])

    return X_train_pool, y_train_pool, X_test, y_test


# ==============================================================================
# Public API
# ==============================================================================

#: Maps dataset name → function that loads ALL trials for one subject
_LOADER_ALL: Dict[str, callable] = {
    "bcic_iv_2a": _load_bcic_iv_2a,
    "bcic_iv_2b": _load_bcic_iv_2b,
    "bcic_iii_4a": _load_bcic_iii_4a,
    "wbcic_shu": _load_wbcic_shu,
}

#: Maps dataset name → function that loads train_pool and test
_LOADER_TE: Dict[str, callable] = {
    "bcic_iv_2a": _load_bcic_iv_2a_train_test,
    "bcic_iv_2b": _load_bcic_iv_2b_train_test,
    "bcic_iii_4a": _load_bcic_iii_4a_train_test,
    "wbcic_shu": _load_wbcic_shu_train_test
}


def load_subject_data(
        data_root: str,
        dataset_name: str,
        subject: int,
) -> Tuple[np.ndarray, np.ndarray]:
    key = dataset_name.strip().lower()
    if key not in _LOADER_ALL:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported: {list(_LOADER_ALL.keys())}"
        )
    return _LOADER_ALL[key](data_root, subject)


def load_subject_train_test(
        data_root: str,
        dataset_name: str,
        subject: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = dataset_name.strip().lower()
    if key not in _LOADER_TE:
        raise ValueError(
            f"Train/test split not available for '{dataset_name}'. "
            f"Supported: {list(_LOADER_TE.keys())}"
        )
    return _LOADER_TE[key](data_root, subject)


# ==============================================================================
# Deterministic stratified split
# ==============================================================================

def deterministic_train_val_test_split(
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train / val / test split using pure NumPy.

    Splitting is per-class to preserve label proportions.
    ``np.random.RandomState(seed)`` guarantees identical results across any environment for the same seed.

    Args:
        X: Feature array  [N, ...]
        y: Label array    [N,]
        train_ratio, val_ratio, test_ratio: Split proportions (sum = 1.0).
        seed: Random seed for shuffling.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    n_total = len(y)

    train_idx: List[np.ndarray] = []
    val_idx: List[np.ndarray] = []
    test_idx: List[np.ndarray] = []

    for cls in classes:
        indices = np.where(y == cls)[0]
        rng.shuffle(indices)
        n_cls = len(indices)

        n_train = int(n_cls * train_ratio)
        n_val = int(n_cls * val_ratio)

        # Guard: ensure ≥ 1 sample in val when test_ratio > 0 and enough samples
        if test_ratio > 0 and n_val == 0 and n_cls >= 3:
            n_val = 1
        if test_ratio > 0 and n_train + n_val >= n_cls and n_cls >= 3:
            n_train = n_cls - 2
            n_val = 1

        n_test = n_cls - n_train - n_val

        # When test_ratio == 0: absorb remainder into train (no wasted samples)
        if test_ratio == 0.0 and n_test > 0:
            n_train += n_test
            n_test = 0

        train_idx.append(indices[:n_train])
        val_idx.append(indices[n_train:n_train + n_val])
        test_idx.append(indices[n_train + n_val:])

    all_train = np.concatenate(train_idx)
    all_val = np.concatenate(val_idx)
    all_test = np.concatenate(test_idx)

    # Log
    print(
        f"Split (seed={seed}): "
        f"Train={len(all_train)} ({len(all_train) / n_total:.0%}), "
        f"Val={len(all_val)} ({len(all_val) / n_total:.0%}), "
        f"Test={len(all_test)} ({len(all_test) / n_total:.0%})"
    )
    for lbl in classes:
        print(
            f"  Class {lbl}: Train={np.sum(y[all_train] == lbl)}, "
            f"Val={np.sum(y[all_val] == lbl)}, Test={np.sum(y[all_test] == lbl)}"
        )

    return X[all_train], y[all_train], X[all_val], y[all_val], X[all_test], y[all_test]


# ==============================================================================
# Leave-One-Subject-Out  (LOSO)  cross-subject loading
# ==============================================================================

def load_loso_data(
        data_root: str,
        dataset_name: str,
        test_subject: int,
        all_subjects: List[int],
        val_subject: Optional[int] = None,
        split_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Leave-One-Subject-Out cross-subject data loading.

    If val_subject is None, 25 % of the training pool is split off as validation using deterministic stratified sampling.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    if test_subject not in all_subjects:
        raise ValueError(f"test_subject {test_subject} not in {all_subjects}")
    if val_subject is not None:
        if val_subject not in all_subjects:
            raise ValueError(f"val_subject {val_subject} not in {all_subjects}")
        if val_subject == test_subject:
            print(f"Note: val_subject ({val_subject}) == test_subject "
                  f"({test_subject}), falling back to stratified validation split.")
            val_subject = None

    # --- test ---
    _, _, X_test, y_test = load_subject_train_test(data_root, dataset_name, test_subject)

    # --- training pool ---
    excluded = {test_subject}
    if val_subject is not None:
        excluded.add(val_subject)
    train_subjects = [s for s in all_subjects if s not in excluded]

    if not train_subjects:
        raise ValueError(f"No training subjects left after excluding {excluded}")

    parts_X, parts_y = [], []
    for s in train_subjects:
        Xs, ys = load_subject_data(data_root, dataset_name, s)
        parts_X.append(Xs)
        parts_y.append(ys)
    X_pool = np.concatenate(parts_X, axis=0)
    y_pool = np.concatenate(parts_y, axis=0)

    # --- validation ---
    if val_subject is not None:
        X_val, y_val = load_subject_data(data_root, dataset_name, val_subject)
        X_train, y_train = X_pool, y_pool
    else:
        X_train, y_train, X_val, y_val, _, _ = deterministic_train_val_test_split(
            X_pool, y_pool,
            train_ratio=0.75, val_ratio=0.25, test_ratio=0.0,
            seed=split_seed,
        )

    vdesc = f"Sub {val_subject}" if val_subject is not None else "stratified"
    print(
        f"LOSO: Test=Sub {test_subject} ({len(y_test)}), "
        f"Val={vdesc} ({len(y_val)}), "
        f"Train={len(train_subjects)} subjects ({len(y_train)})"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_loso_all_test_subjects(
        data_root: str,
        dataset_name: str,
        all_subjects: List[int],
        val_subject: Optional[int] = None,
        split_seed: int = 42,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Run LOSO for every subject as the test subject.

    Returns:
        Dict: test_subject → (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    results = {}
    for test_subj in all_subjects:
        cur_val = val_subject
        if val_subject is not None and val_subject == test_subj:
            print(f"Note: val_subject={val_subject} == test_subject={test_subj}, "
                  f"falling back to stratified validation split.")
            cur_val = None
        results[test_subj] = load_loso_data(
            data_root=data_root,
            dataset_name=dataset_name,
            test_subject=test_subj,
            all_subjects=all_subjects,
            val_subject=cur_val,
            split_seed=split_seed,
        )
    return results


# ==============================================================================
# Self-test
# ==============================================================================

if __name__ == "__main__":
    _data_root = os.path.join("../datasets")
    SPLIT_SEED = 42

    # ---- Within-subject ----
    print("=" * 70)
    print("WITHIN-SUBJECT")
    print("=" * 70)

    within_configs = [
        ("bcic_iv_2a", list(range(1, 10))),
        ("bcic_iv_2b", list(range(1, 10))),
        ("bcic_iii_4a", list(range(1, 6))),
        ("wbcic_shu", list(range(1, 53))),
    ]

    for ds_name, subjects in within_configs:
        for sid in subjects:
            print(f"\n--- {ds_name}, Subject {sid} ---")
            X_pool, y_pool, X_test, y_test = load_subject_train_test(
                _data_root, ds_name, sid
            )
            print(f"  Train pool: {X_pool.shape}, Test: {X_test.shape}")
            print(f"  Train labels: {np.unique(y_pool, return_counts=True)}")
            print(f"  Test labels: {np.unique(y_test, return_counts=True)}")

            # Split training pool → 80 % train, 20 % val
            X_tr, y_tr, X_val, y_val, _, _ = deterministic_train_val_test_split(
                X_pool, y_pool,
                train_ratio=0.8, val_ratio=0.2, test_ratio=0.0,
                seed=SPLIT_SEED,
            )
            print(f"  → Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ---- Cross-subject LOSO ----
    print("\n" + "=" * 70)
    print("CROSS-SUBJECT LOSO")
    print("=" * 70)

    loso_configs = [
        ("bcic_iv_2a", list(range(1, 10)), 9),
        ("bcic_iv_2b", list(range(1, 10)), 9),
        ("bcic_iii_4a", list(range(1, 6)), 5),
        ("wbcic_shu", list(range(1, 53)), 52),
    ]

    for ds_name, subjects, val_subj in loso_configs:
        print(f"\n--- {ds_name} (val_subject={val_subj}) ---")
        results = load_loso_all_test_subjects(
            _data_root, ds_name, subjects, val_subject=val_subj, split_seed=SPLIT_SEED
        )
        for ts, (Xtr, ytr, Xv, yv, Xte, yte) in results.items():
            print(f"  Test S{ts}: Train={Xtr.shape}, Val={Xv.shape}, Test={Xte.shape}")
