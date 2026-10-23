"""
Build a participant-disjoint, 20 Hz, 3-second, 6-channel HAR dataset.

Set the three input roots below to the extracted dataset directories before
running this file. The pipeline performs timestamp/index-based linear
interpolation only for resampling; it does not apply signal-conditioning
filters or remove gravity. Windows never overlap and never cross a source
activity/trial boundary.

UCI-HAR is audited but deliberately excluded: the supplied standard release
contains pre-windowed (128 samples at 50 Hz) signals, not continuous recordings
from which valid 60-sample windows can be reconstructed.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# Configure these paths. Relative paths are resolved from this script's folder.
SCRIPT_DIR = Path(__file__).resolve().parent
WISDM_ROOT = SCRIPT_DIR / "wisdm-dataset"
MOTIONSENSE_ROOT = SCRIPT_DIR / "motionsense-dataset"
UCI_HAR_ROOT = SCRIPT_DIR / "uci-dataset"
OUTPUT_ROOT = SCRIPT_DIR / "processed_dataset"

# WISDM has phone and watch copies of the same participants. Select one device
# location to avoid treating the same person/device streams as separate data.
WISDM_DEVICE = "phone"  # "phone" or "watch"

# MotionSense CSVs contain a sample index, not timestamps. The supplied files
# do not encode a sampling interval; this is the documented nominal acquisition
# rate and is intentionally explicit/configurable rather than inferred.
MOTIONSENSE_SOURCE_HZ = 50.0

TARGET_HZ = 20
WINDOW_SECONDS = 3
WINDOW_SAMPLES = TARGET_HZ * WINDOW_SECONDS
RANDOM_STATE = 42
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
CLASS_NAMES = {
    "A": "Walking",
    "B": "Jogging",
    "D": "Sitting",
    "E": "Standing",
}
WISDM_ACTIVITY_MAP = {
    "A": ("Walking", "A"),
    "B": ("Jogging", "B"),
    "D": ("Sitting", "D"),
    "E": ("Standing", "E"),
}
MOTIONSENSE_ACTIVITY_MAP = {
    "wlk": ("Walking", "A"),
    "jog": ("Jogging", "B"),
    "sit": ("Sitting", "D"),
    "std": ("Standing", "E"),
}
DATASET_PREFIXES = {
    "WISDM": "WISDM",
    "MotionSense": "MotionSense",
    "UCI_HAR": "UCI",
}
TIMESTAMP_STEP_NS = 1_000_000_000 // TARGET_HZ


@dataclass(frozen=True)
class RawRecord:
    subject_id: str
    activity_code: str
    timestamp_ns: int
    xyz: tuple[float, float, float]


@dataclass
class SignalSegment:
    activity_code: str
    timestamps_ns: np.ndarray
    values: np.ndarray


@dataclass
class WindowRecord:
    client_id: str
    dataset: str
    original_subject_id: str
    activity: str
    label: str
    values: np.ndarray
    source_file: str
    window_index: int


def warn(issues: list[str], path: Path | str, message: str) -> None:
    text = f"{path}: {message}"
    issues.append(text)
    print(f"WARNING: {text}")


def subject_sort_key(subject_id: str) -> tuple[int, int | str]:
    if subject_id.isdigit():
        return (0, int(subject_id))
    return (1, subject_id)


def make_client_ids(dataset: str, subject_ids: Iterable[str]) -> dict[str, str]:
    prefix = DATASET_PREFIXES[dataset]
    ordered = sorted(set(subject_ids), key=subject_sort_key)
    return {subject: f"{prefix}_{index:03d}" for index, subject in enumerate(ordered, 1)}


def relative_source(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def add_windows(
    records: list[WindowRecord],
    values: np.ndarray,
    *,
    dataset: str,
    subject_id: str,
    client_id: str,
    activity: str,
    label: str,
    source_file: str,
) -> int:
    if values.ndim != 2 or values.shape[1] != len(CHANNELS):
        raise ValueError(f"Expected resampled data shaped (time, 6), got {values.shape}")
    count = len(values) // WINDOW_SAMPLES
    for window_index in range(count):
        start = window_index * WINDOW_SAMPLES
        window = np.asarray(
            values[start : start + WINDOW_SAMPLES],
            dtype=np.float32,
        )
        if window.shape != (WINDOW_SAMPLES, len(CHANNELS)):
            raise AssertionError(f"Invalid window shape: {window.shape}")
        records.append(
            WindowRecord(
                client_id=client_id,
                dataset=dataset,
                original_subject_id=subject_id,
                activity=activity,
                label=label,
                values=window,
                source_file=source_file,
                window_index=window_index,
            )
        )
    return count


def parse_wisdm_file(
    path: Path,
    expected_subject_id: str,
    issues: list[str],
    source_rows: Counter[str],
) -> list[RawRecord] | None:
    parsed: list[RawRecord] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            for line_number, row in enumerate(reader, 1):
                if not row or all(not field.strip() for field in row):
                    continue
                row = [field.strip() for field in row]
                row[-1] = row[-1].rstrip(";").strip()
                if len(row) != 6:
                    raise ValueError(f"line {line_number} has {len(row)} fields, expected 6")
                subject_id, activity_code = row[0], row[1].upper()
                if subject_id != expected_subject_id:
                    raise ValueError(
                        f"line {line_number} has subject {subject_id}, "
                        f"but filename identifies {expected_subject_id}"
                    )
                timestamp_ns = int(row[2])
                xyz = (float(row[3]), float(row[4]), float(row[5]))
                if not all(math.isfinite(value) for value in xyz):
                    raise ValueError(f"line {line_number} contains a non-finite sensor value")
                parsed.append(
                    RawRecord(subject_id, activity_code, timestamp_ns, xyz)
                )
                source_rows[activity_code] += 1
    except (OSError, csv.Error, ValueError) as exc:
        warn(issues, path, f"file could not be processed ({exc}); entire file excluded")
        return None

    if not parsed:
        warn(issues, path, "file contains no usable sensor rows")
        return None

    for previous, current in zip(parsed, parsed[1:]):
        if (
            current.activity_code == previous.activity_code
            and current.timestamp_ns <= previous.timestamp_ns
        ):
            warn(
                issues,
                path,
                "timestamps are duplicated or non-increasing within an activity "
                "run; entire file excluded",
            )
            return None
    return parsed


def make_wisdm_segments(records: list[RawRecord]) -> tuple[list[SignalSegment], float | None]:
    if len(records) < 2:
        return [], None
    within_activity_deltas = [
        right.timestamp_ns - left.timestamp_ns
        for left, right in zip(records, records[1:])
        if right.activity_code == left.activity_code
        and right.timestamp_ns > left.timestamp_ns
    ]
    if not within_activity_deltas:
        return [], None
    median_delta_ns = float(np.median(within_activity_deltas))
    if median_delta_ns <= 0:
        return [], None
    gap_limit_ns = max(1.75 * median_delta_ns, 120_000_000.0)

    segments: list[SignalSegment] = []
    current: list[RawRecord] = []
    for record in records:
        if current:
            previous = current[-1]
            activity_changed = record.activity_code != previous.activity_code
            gap_found = record.timestamp_ns - previous.timestamp_ns > gap_limit_ns
            if activity_changed or gap_found:
                segments.append(_records_to_segment(current))
                current = []
        current.append(record)
    if current:
        segments.append(_records_to_segment(current))
    return segments, 1_000_000_000.0 / median_delta_ns


def _records_to_segment(records: list[RawRecord]) -> SignalSegment:
    return SignalSegment(
        activity_code=records[0].activity_code,
        timestamps_ns=np.asarray(
            [record.timestamp_ns for record in records], dtype=np.int64
        ),
        values=np.asarray([record.xyz for record in records], dtype=np.float64),
    )


def interpolate_wisdm_segment(
    segment: SignalSegment, target_ns: np.ndarray
) -> np.ndarray:
    source_seconds = (
        segment.timestamps_ns - segment.timestamps_ns[0]
    ).astype(np.float64) / 1_000_000_000.0
    target_seconds = (
        target_ns - segment.timestamps_ns[0]
    ).astype(np.float64) / 1_000_000_000.0
    return np.column_stack(
        [
            np.interp(target_seconds, source_seconds, segment.values[:, axis])
            for axis in range(3)
        ]
    )


def _wisdm_file_map(
    root: Path,
    sensor: str,
    device: str,
    issues: list[str],
    *,
    required: bool,
) -> dict[str, Path]:
    matches = sorted(
        path
        for path in root.rglob("*.txt")
        if path.is_file()
        and path.parent.name == sensor
        and path.parent.parent.name == device
    )
    if not matches:
        if required:
            raise FileNotFoundError(
                f"No WISDM {sensor} files found under {root} for device {device!r}"
            )
        return {}
    result: dict[str, Path] = {}
    for path in matches:
        match = re.fullmatch(rf"data_(\d+)_{sensor}_{device}\.txt", path.name)
        if not match:
            warn(
                issues,
                path,
                "text file in selected sensor folder did not match the expected "
                "participant filename and was excluded",
            )
            continue
        subject_id = match.group(1)
        if subject_id in result:
            raise ValueError(
                f"Multiple WISDM {sensor} files found for participant {subject_id}: "
                f"{result[subject_id]} and {path}. Point WISDM_ROOT at one extracted copy."
            )
        result[subject_id] = path
    return result


def load_wisdm(
    root: Path,
    issues: list[str],
    source_activity_rows: dict[str, Counter[str]],
) -> tuple[list[WindowRecord], list[str], list[float]]:
    if WISDM_DEVICE not in {"phone", "watch"}:
        raise ValueError("WISDM_DEVICE must be 'phone' or 'watch'")
    accel_files = _wisdm_file_map(
        root, "accel", WISDM_DEVICE, issues, required=True
    )
    gyro_files = _wisdm_file_map(
        root, "gyro", WISDM_DEVICE, issues, required=False
    )
    subjects = sorted(set(accel_files) | set(gyro_files), key=subject_sort_key)
    client_ids = make_client_ids("WISDM", subjects)
    records: list[WindowRecord] = []
    rates: list[float] = []
    source_activity_rows["WISDM"] = Counter()

    print("\nDATASET AUDIT - WISDM")
    print(
        f"  Structure: separate raw accel/gyro files under a selected "
        f"{WISDM_DEVICE!r} device directory; {len(accel_files)} accel and "
        f"{len(gyro_files)} gyro files found."
    )
    print(
        "  Timestamp: numeric nanoseconds from the supplied raw rows; preserve "
        "each activity run because timestamps reset across activity sessions."
    )
    print("  Target sensor channels: XYZ accelerometer + XYZ gyroscope.")
    print("  Selected activities: activity_key.txt codes A, B, D, E.")
    print(f"  Participant IDs: {len(subjects)} distinct IDs from filenames and rows.")

    for subject_id in subjects:
        if subject_id not in accel_files or subject_id not in gyro_files:
            missing = "accelerometer" if subject_id not in accel_files else "gyroscope"
            warn(
                issues,
                root,
                f"WISDM participant {subject_id} has no {missing} file; "
                "cannot enter the 6-channel experiment",
            )
            continue
        accel_rows = parse_wisdm_file(
            accel_files[subject_id],
            subject_id,
            issues,
            source_activity_rows["WISDM"],
        )
        gyro_rows = parse_wisdm_file(
            gyro_files[subject_id],
            subject_id,
            issues,
            Counter(),
        )
        if accel_rows is None or gyro_rows is None:
            continue
        accel_segments, accel_rate = make_wisdm_segments(accel_rows)
        gyro_segments, gyro_rate = make_wisdm_segments(gyro_rows)
        rates.extend(rate for rate in (accel_rate, gyro_rate) if rate is not None)

        has_overlapping_target_data = False
        for accel_segment in accel_segments:
            mapped = WISDM_ACTIVITY_MAP.get(accel_segment.activity_code)
            if mapped is None:
                continue
            activity, label = mapped
            matching_gyro = [
                segment
                for segment in gyro_segments
                if segment.activity_code == accel_segment.activity_code
            ]
            for gyro_segment in matching_gyro:
                overlap_start = max(
                    int(accel_segment.timestamps_ns[0]),
                    int(gyro_segment.timestamps_ns[0]),
                )
                overlap_end = min(
                    int(accel_segment.timestamps_ns[-1]),
                    int(gyro_segment.timestamps_ns[-1]),
                )
                if overlap_end < overlap_start:
                    continue
                count = (overlap_end - overlap_start) // TIMESTAMP_STEP_NS + 1
                if count < WINDOW_SAMPLES:
                    continue
                target_ns = (
                    overlap_start
                    + np.arange(count, dtype=np.int64) * TIMESTAMP_STEP_NS
                )
                accel = interpolate_wisdm_segment(accel_segment, target_ns)
                gyro = interpolate_wisdm_segment(gyro_segment, target_ns)
                combined = np.column_stack((accel, gyro))
                source = (
                    f"accel:{relative_source(accel_files[subject_id], root)}|"
                    f"gyro:{relative_source(gyro_files[subject_id], root)}"
                )
                made = add_windows(
                    records,
                    combined,
                    dataset="WISDM",
                    subject_id=subject_id,
                    client_id=client_ids[subject_id],
                    activity=activity,
                    label=label,
                    source_file=source,
                )
                has_overlapping_target_data |= made > 0
        if not has_overlapping_target_data:
            warn(
                issues,
                root,
                f"WISDM participant {subject_id} has no complete aligned "
                "3-second target-activity windows across both modalities",
            )

    if rates:
        print(
            "  Measured approximate source rate (median of within-activity "
            f"timestamp deltas): {np.median(rates):.3f} Hz."
        )
    else:
        print("  Measured approximate source rate: unavailable (no valid intervals).")
    print(
        "  Resampling: actual timestamp overlap, uniform 20 Hz target grid, "
        "linear interpolation; no missing gyro values are filled."
    )
    return records, subjects, rates


def locate_motionsense_files(root: Path) -> list[Path]:
    files = sorted(
        path
        for path in root.rglob("*.csv")
        if path.is_file()
        and path.name != "data_subjects_info.csv"
        and not path.name.startswith("._")
        and "__MACOSX" not in path.parts
    )
    if not files:
        raise FileNotFoundError(f"No MotionSense sub_*.csv files found under {root}")
    return files


def load_motionsense(
    root: Path,
    issues: list[str],
    source_activity_rows: dict[str, Counter[str]],
) -> tuple[list[WindowRecord], list[str]]:
    if not math.isfinite(MOTIONSENSE_SOURCE_HZ) or MOTIONSENSE_SOURCE_HZ <= 0:
        raise ValueError("MOTIONSENSE_SOURCE_HZ must be a finite positive number")
    files = locate_motionsense_files(root)
    file_info: list[tuple[Path, str, str]] = []
    subjects: set[str] = set()
    trial_folders: set[str] = set()
    for path in files:
        folder_name = path.parent.name
        match = re.fullmatch(r"([a-zA-Z]+)_\d+", folder_name)
        subject_match = re.fullmatch(r"sub_(\d+)\.csv", path.name)
        if not match or not subject_match:
            warn(issues, path, "file path does not match activity_trial/sub_ID.csv")
            continue
        activity_code = match.group(1).lower()
        subject_id = subject_match.group(1)
        subjects.add(subject_id)
        trial_folders.add(folder_name)
        file_info.append((path, activity_code, subject_id))
    client_ids = make_client_ids("MotionSense", subjects)
    source_activity_rows["MotionSense"] = Counter()
    records: list[WindowRecord] = []

    print("\nDATASET AUDIT - MotionSense")
    print(
        f"  Structure: activity/trial folders containing sub_ID.csv; "
        f"{len(file_info)} structurally valid CSV files in "
        f"{len(trial_folders)} trial folders."
    )
    print(
        "  Timestamp: no elapsed-time column; first CSV field is a sample index. "
        f"Using the explicit configured nominal rate {MOTIONSENSE_SOURCE_HZ:g} Hz."
    )
    print(
        "  Sensors: userAcceleration XYZ + gravity XYZ for acceleration; "
        "rotationRate XYZ for gyroscope."
    )
    print(
        "  Folder prefixes: wlk=Walking, jog=Jogging, sit=Sitting, std=Standing; "
        "stairs (ups/dws) and all other activities are excluded."
    )
    print(f"  Participants: {len(subjects)} distinct sub_ID values.")

    expected_columns = (
        "gravity.x",
        "gravity.y",
        "gravity.z",
        "rotationRate.x",
        "rotationRate.y",
        "rotationRate.z",
        "userAcceleration.x",
        "userAcceleration.y",
        "userAcceleration.z",
    )
    for path, activity_code, subject_id in file_info:
        mapped = MOTIONSENSE_ACTIVITY_MAP.get(activity_code)
        if mapped is None:
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    row_count = max(sum(1 for _ in handle) - 1, 0)
                source_activity_rows["MotionSense"][activity_code] += row_count
            except OSError as exc:
                warn(issues, path, f"could not count excluded-activity rows ({exc})")
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                headers = reader.fieldnames or []
                index_header = headers[0].strip().lower() if headers else None
                if index_header not in {"", "unnamed: 0"}:
                    raise ValueError(
                        "first column is not the observed unnamed sample-index column"
                    )
                missing = [column for column in expected_columns if column not in headers]
                if missing:
                    raise ValueError(f"missing required columns: {', '.join(missing)}")
                sample_indices: list[int] = []
                numeric_rows: list[list[float]] = []
                row_count = 0
                for row_count, row in enumerate(reader, 1):
                    sample_index = int(row[headers[0]])
                    values = [float(row[column]) for column in expected_columns]
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError(f"data row {row_count} contains non-finite values")
                    sample_indices.append(sample_index)
                    numeric_rows.append(values)
        except (OSError, csv.Error, ValueError, TypeError) as exc:
            warn(issues, path, f"file could not be processed ({exc}); entire file excluded")
            continue
        if len(sample_indices) < WINDOW_SAMPLES:
            warn(
                issues,
                path,
                f"only {len(sample_indices)} rows; fewer than {WINDOW_SAMPLES} target samples",
            )
            continue
        indices = np.asarray(sample_indices, dtype=np.int64)
        if np.any(np.diff(indices) != 1):
            warn(
                issues,
                path,
                "sample index is not strictly consecutive; file excluded rather than "
                "interpolating across missing/duplicated source rows",
            )
            continue
        matrix = np.asarray(numeric_rows, dtype=np.float64)
        values_by_name = {name: matrix[:, i] for i, name in enumerate(expected_columns)}
        acceleration = np.column_stack(
            [
                values_by_name[f"userAcceleration.{axis}"]
                + values_by_name[f"gravity.{axis}"]
                for axis in "xyz"
            ]
        )
        gyro = np.column_stack(
            [values_by_name[f"rotationRate.{axis}"] for axis in "xyz"]
        )
        source = np.column_stack((acceleration, gyro))

        source_time = (indices - indices[0]).astype(np.float64) / MOTIONSENSE_SOURCE_HZ
        target_count = (
            int(math.floor(source_time[-1] * TARGET_HZ + 1e-10)) + 1
        )
        target_time = np.arange(target_count, dtype=np.float64) / TARGET_HZ
        resampled = np.column_stack(
            [
                np.interp(target_time, source_time, source[:, channel])
                for channel in range(source.shape[1])
            ]
        )
        activity, label = mapped
        source_activity_rows["MotionSense"][activity_code] += len(indices)
        add_windows(
            records,
            resampled,
            dataset="MotionSense",
            subject_id=subject_id,
            client_id=client_ids[subject_id],
            activity=activity,
            label=label,
            source_file=relative_source(path, root),
        )

    print(
        "  Resampling: sample-index timing at configured source rate to 20 Hz "
        "by linear interpolation; acceleration includes gravity."
    )
    return records, sorted(subjects, key=subject_sort_key)


def locate_uci_directory(root: Path) -> Path:
    candidates = [root, root / "UCI HAR Dataset"]
    candidates.extend(
        path for path in root.glob("*") if path.is_dir() and path.name == "UCI HAR Dataset"
    )
    candidates.extend(
        inertial_dir.parent.parent
        for inertial_dir in root.rglob("Inertial Signals")
        if inertial_dir.is_dir()
    )
    for candidate in candidates:
        if (
            (candidate / "train" / "Inertial Signals").is_dir()
            and (candidate / "test" / "Inertial Signals").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        f"Could not locate standard UCI-HAR train/test Inertial Signals under {root}"
    )


def inspect_uci_vector_file(path: Path, expected_rows: int) -> int:
    row_count = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            width = len(line.split())
            if width != 128:
                raise ValueError(
                    f"line {line_number} has {width} values; expected 128 "
                    "pre-windowed samples"
                )
            row_count += 1
    if row_count != expected_rows:
        raise ValueError(f"has {row_count} rows; expected {expected_rows}")
    return row_count


def load_uci_har(
    root: Path, issues: list[str]
) -> tuple[list[WindowRecord], dict[str, object]]:
    base = locate_uci_directory(root)
    subject_ids: set[str] = set()
    raw_class_windows: Counter[str] = Counter()
    split_rows: dict[str, int] = {}
    usable_sensor_files: list[str] = []
    for split in ("train", "test"):
        subject_file = base / split / f"subject_{split}.txt"
        label_file = base / split / f"y_{split}.txt"
        subjects = np.loadtxt(subject_file, dtype=int, ndmin=1)
        labels = np.loadtxt(label_file, dtype=int, ndmin=1)
        if len(subjects) != len(labels):
            raise ValueError(
                f"UCI-HAR {split}: subject/label row counts do not match "
                f"({len(subjects)} vs {len(labels)})"
            )
        split_rows[split] = len(labels)
        subject_ids.update(str(int(value)) for value in subjects)
        for value in labels:
            raw_class_windows[str(int(value))] += 1
        inertial = base / split / "Inertial Signals"
        for sensor_axis in (
            "total_acc_x",
            "total_acc_y",
            "total_acc_z",
            "body_gyro_x",
            "body_gyro_y",
            "body_gyro_z",
        ):
            path = inertial / f"{sensor_axis}_{split}.txt"
            rows = inspect_uci_vector_file(path, len(labels))
            usable_sensor_files.append(str(path))
            if rows != len(labels):
                raise AssertionError("UCI-HAR inertial row validation failed")

    labels_to_activity = {
        "1": ("Walking", "A"),
        "4": ("Sitting", "D"),
        "5": ("Standing", "E"),
    }
    target_raw_windows = {
        label: raw_class_windows.get(source_label, 0)
        for source_label, (_, label) in labels_to_activity.items()
    }
    info: dict[str, object] = {
        "subjects": sorted(subject_ids, key=subject_sort_key),
        "source_windows": sum(split_rows.values()),
        "split_rows": split_rows,
        "raw_class_windows": dict(raw_class_windows),
        "target_raw_windows": target_raw_windows,
        "sensor_files": len(usable_sensor_files),
    }

    print("\nDATASET AUDIT - UCI-HAR")
    print(
        f"  Structure: standard train/test release; {sum(split_rows.values())} "
        "labeled records and six per-axis inertial files per split."
    )
    print(
        "  Source sampling rate: 50 Hz; each supplied inertial row has 128 values "
        "(2.56 seconds)."
    )
    print(
        "  Sensors: total acceleration XYZ and gyroscope XYZ are supplied per "
        "pre-windowed record; README documents prior noise filtering."
    )
    print(
        "  Activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, "
        "STANDING, LAYING; no Jogging."
    )
    print(
        f"  Participants: {len(subject_ids)} source subject IDs across the "
        "supplied train/test windows."
    )
    print(
        "  Continuous data: NOT present. Pre-windowed records cannot be joined "
        "to make continuous 3-second windows; UCI-HAR contributes zero samples "
        "to this unified dataset."
    )
    print(f"  Target-class pre-windowed source records (not emitted): {target_raw_windows}")
    return [], info


def split_counts(n_clients: int) -> dict[str, int]:
    names = tuple(SPLIT_RATIOS)
    exact = {name: n_clients * SPLIT_RATIOS[name] for name in names}
    counts = {name: math.floor(exact[name]) for name in names}
    remainder = n_clients - sum(counts.values())
    order = sorted(names, key=lambda name: (exact[name] - counts[name], -names.index(name)), reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def assign_client_splits(records: list[WindowRecord]) -> dict[str, str]:
    clients_by_dataset: dict[str, set[str]] = defaultdict(set)
    for record in records:
        clients_by_dataset[record.dataset].add(record.client_id)
    rng = np.random.default_rng(RANDOM_STATE)
    assignment: dict[str, str] = {}
    for dataset in sorted(clients_by_dataset):
        clients = sorted(clients_by_dataset[dataset])
        shuffled = list(rng.permutation(clients))
        counts = split_counts(len(shuffled))
        offset = 0
        for split, count in counts.items():
            for client_id in shuffled[offset : offset + count]:
                assignment[client_id] = split
            offset += count
    return assignment


def validate_records(
    records: list[WindowRecord], client_splits: dict[str, str]
) -> None:
    if TARGET_HZ != 20 or WINDOW_SECONDS != 3 or WINDOW_SAMPLES != 60:
        raise AssertionError("Configured output must be 20 Hz, 3 seconds, 60 samples")
    if not records:
        raise RuntimeError(
            "No valid 6-channel windows were produced. Verify input paths, "
            "source files, and WISDM gyro availability."
        )
    client_identity: dict[str, tuple[str, str]] = {}
    for record in records:
        if record.values.shape != (60, 6):
            raise AssertionError(f"{record.client_id}: invalid sample shape {record.values.shape}")
        if record.label not in CLASS_NAMES:
            raise AssertionError(f"Unexpected label {record.label!r}")
        if not np.isfinite(record.values).all():
            raise ValueError(
                f"Non-finite sensor value in {record.source_file}, "
                f"client {record.client_id}"
            )
        identity = (record.dataset, record.original_subject_id)
        previous = client_identity.setdefault(record.client_id, identity)
        if previous != identity:
            raise AssertionError(f"Client ID collision: {record.client_id} maps to {previous} and {identity}")
        if record.client_id not in client_splits:
            raise AssertionError(f"No split assigned for client {record.client_id}")
    split_clients: dict[str, set[str]] = defaultdict(set)
    for client_id, split in client_splits.items():
        split_clients[split].add(client_id)
    for left in ("train", "validation", "test"):
        for right in ("train", "validation", "test"):
            if left < right and split_clients[left] & split_clients[right]:
                raise AssertionError(f"Client leakage between {left} and {right}")
    if set(client_splits.values()) - set(SPLIT_RATIOS):
        raise AssertionError("Unknown client split was produced")


def feature_columns() -> list[str]:
    return [
        f"{channel}_{time_index}"
        for channel in CHANNELS
        for time_index in range(1, WINDOW_SAMPLES + 1)
    ]


def write_split(
    output_root: Path,
    split: str,
    records: list[WindowRecord],
) -> None:
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    if records:
        x = np.stack([record.values for record in records]).astype(np.float32, copy=False)
    else:
        x = np.empty((0, WINDOW_SAMPLES, len(CHANNELS)), dtype=np.float32)
    y = np.asarray([record.label for record in records], dtype="<U1")
    client_ids = np.asarray([record.client_id for record in records], dtype=str)
    if x.shape != (len(records), 60, 6):
        raise AssertionError(f"{split}: saved tensor shape would be {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{split}: X contains NaN or infinite values")
    if set(y.tolist()) - set(CLASS_NAMES):
        raise AssertionError(f"{split}: unexpected label value")

    np.save(split_dir / "X.npy", x)
    np.save(split_dir / "y.npy", y)
    np.save(split_dir / "client_ids.npy", client_ids)

    metadata_path = split_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "client_id",
            "dataset",
            "original_subject_id",
            "activity",
            "label",
            "source_file",
            "window_index",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "client_id": record.client_id,
                    "dataset": record.dataset,
                    "original_subject_id": record.original_subject_id,
                    "activity": record.activity,
                    "label": record.label,
                    "source_file": record.source_file,
                    "window_index": record.window_index,
                }
            )

    csv_path = output_root / f"fl_{split}.csv"
    csv_fields = [
        "client_id",
        "dataset",
        "original_subject_id",
        "label",
        *feature_columns(),
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(csv_fields)
        for record in records:
            # Channel-major order matches acc_x_1..60, then acc_y_1..60, etc.
            flattened = record.values.T.reshape(-1)
            writer.writerow(
                [
                    record.client_id,
                    record.dataset,
                    record.original_subject_id,
                    record.label,
                    *flattened.tolist(),
                ]
            )

    saved = np.load(split_dir / "X.npy", mmap_mode="r")
    if saved.shape != (len(records), 60, 6):
        raise AssertionError(f"{split}: saved X.npy has invalid shape {saved.shape}")


def class_counts(records: Iterable[WindowRecord]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts[record.label] += 1
    return counts


def write_summary(
    output_root: Path,
    records: list[WindowRecord],
    client_splits: dict[str, str],
    uci_info: dict[str, object],
    source_activity_rows: dict[str, Counter[str]],
) -> None:
    columns = [
        "scope",
        "dataset",
        "split",
        "client_id",
        "original_subject_id",
        "n_clients",
        "n_samples",
        "A_samples",
        "B_samples",
        "D_samples",
        "E_samples",
        "status",
        "details",
    ]
    rows: list[dict[str, object]] = []
    datasets = ("WISDM", "MotionSense", "UCI_HAR")

    def append_row(
        scope: str,
        dataset: str,
        split: str = "",
        client_id: str = "",
        original_subject_id: str = "",
        subset: list[WindowRecord] | None = None,
        status: str = "",
        details: str = "",
    ) -> None:
        subset = subset or []
        counts = class_counts(subset)
        n_clients = len({item.client_id for item in subset})
        rows.append(
            {
                "scope": scope,
                "dataset": dataset,
                "split": split,
                "client_id": client_id,
                "original_subject_id": original_subject_id,
                "n_clients": n_clients,
                "n_samples": len(subset),
                **{f"{label}_samples": counts.get(label, 0) for label in CLASS_NAMES},
                "status": status,
                "details": details,
            }
        )

    for dataset in datasets:
        dataset_records = [record for record in records if record.dataset == dataset]
        details = ""
        status = "included"
        if dataset == "UCI_HAR":
            status = "excluded_no_continuous_raw_signal"
            details = (
                f"source_windows={uci_info.get('source_windows', 0)};"
                f"target_pre_windowed_records={uci_info.get('target_raw_windows', {})};"
                "not re-windowed"
            )
        else:
            discarded = {
                code: count
                for code, count in source_activity_rows.get(dataset, Counter()).items()
                if (
                    (dataset == "WISDM" and code not in WISDM_ACTIVITY_MAP)
                    or (dataset == "MotionSense" and code not in MOTIONSENSE_ACTIVITY_MAP)
                )
            }
            details = f"discarded_source_activity_rows={discarded}"
        append_row(
            "dataset",
            dataset,
            subset=dataset_records,
            status=status,
            details=details,
        )
        for split in SPLIT_RATIOS:
            split_records = [
                record
                for record in dataset_records
                if client_splits.get(record.client_id) == split
            ]
            append_row(
                "split",
                dataset,
                split=split,
                subset=split_records,
                status=status,
            )
        for client_id in sorted({record.client_id for record in dataset_records}):
            client_records = [
                record for record in dataset_records if record.client_id == client_id
            ]
            append_row(
                "client",
                dataset,
                split=client_splits[client_id],
                client_id=client_id,
                original_subject_id=client_records[0].original_subject_id,
                subset=client_records,
                status="included",
            )

    uci_subjects = list(uci_info.get("subjects", []))
    uci_client_ids = make_client_ids("UCI_HAR", uci_subjects)
    for subject_id in uci_subjects:
        append_row(
            "source_subject",
            "UCI_HAR",
            client_id=uci_client_ids[subject_id],
            original_subject_id=subject_id,
            status="excluded_no_continuous_raw_signal",
            details="source participant retained in audit; no output windows",
        )

    with (output_root / "dataset_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def print_final_summary(
    records: list[WindowRecord],
    client_splits: dict[str, str],
    source_activity_rows: dict[str, Counter[str]],
    uci_info: dict[str, object],
    issues: list[str],
) -> None:
    print("\n" + "=" * 56)
    print("DATASET SUMMARY")
    print("=" * 56)
    print(f"{'Dataset':<20}{'Clients':>12}{'Samples':>14}")
    print("-" * 46)
    for dataset in ("WISDM", "MotionSense", "UCI_HAR"):
        subset = [record for record in records if record.dataset == dataset]
        clients = {record.client_id for record in subset}
        if dataset == "UCI_HAR":
            print(
                f"{dataset:<20}{len(uci_info.get('subjects', [])):>12}"
                f"{len(subset):>14}  (source clients audited; excluded)"
            )
        else:
            print(f"{dataset:<20}{len(clients):>12}{len(subset):>14}")
    total_clients = len(client_splits)
    print("-" * 46)
    print(f"{'TOTAL included':<20}{total_clients:>12}{len(records):>14}")

    print("\nClass distribution:")
    for label, activity in CLASS_NAMES.items():
        print(f"  {label} = {activity}")
    print("\nClass distribution per dataset (output windows):")
    for dataset in ("WISDM", "MotionSense", "UCI_HAR"):
        counts = class_counts(record for record in records if record.dataset == dataset)
        print(
            f"  {dataset}: "
            + ", ".join(f"{label}={counts.get(label, 0)}" for label in CLASS_NAMES)
        )
    for split in SPLIT_RATIOS:
        selected = [
            record
            for record in records
            if client_splits.get(record.client_id) == split
        ]
        clients = {record.client_id for record in selected}
        counts = class_counts(selected)
        print(f"\n{split.title()}:")
        print(f"  Clients: {len(clients)}")
        print(f"  Samples: {len(selected)}")
        for label in CLASS_NAMES:
            print(f"  {label}: {counts.get(label, 0)}")
    print("\nSamples per client:")
    per_client: dict[tuple[str, str, str], int] = Counter(
        (record.dataset, record.client_id, record.original_subject_id)
        for record in records
    )
    for (dataset, client_id, subject_id), count in sorted(per_client.items()):
        print(
            f"  {client_id} ({dataset}, original_subject_id={subject_id}): "
            f"{count} windows, {client_splits[client_id].upper()}"
        )
    print("\nDiscarded source activities (row/window counts):")
    for dataset, counts in source_activity_rows.items():
        if dataset == "WISDM":
            discarded = {code: count for code, count in counts.items() if code not in WISDM_ACTIVITY_MAP}
        else:
            discarded = {
                code: count
                for code, count in counts.items()
                if code not in MOTIONSENSE_ACTIVITY_MAP
            }
        print(f"  {dataset}: {discarded or '{}'}")
    if uci_info:
        print(
            "  UCI_HAR source class windows: "
            f"{uci_info.get('raw_class_windows', {})}; no UCI windows were emitted."
        )
    print("\nMissing sensor channels:")
    if not any("no gyroscope" in issue or "no accelerometer" in issue for issue in issues):
        print(
            "  None in successfully paired WISDM selected-device streams or "
            "MotionSense."
        )
    else:
        for issue in issues:
            if "no gyroscope" in issue or "no accelerometer" in issue:
                print(f"  {issue}")
    print("\nOutput representation:")
    print("  X.shape per sample: (60, 6)")
    print("  Channels: " + ", ".join(CHANNELS))
    print("  Sampling frequency: 20 Hz")
    print("  Window duration: 3 seconds; 60 samples; no overlap")
    print("  Filtering: NONE; no detrending or gravity removal")
    print("  Resampling: timestamp/index-based linear interpolation only")
    print("  UCI-HAR: excluded because only pre-windowed, already preprocessed signals are supplied")
    print(f"  Fixed random seed: {RANDOM_STATE}; split ratios: {SPLIT_RATIOS}")
    print(f"  Input UCI target pre-windowed rows not used: {uci_info.get('target_raw_windows', {})}")
    if issues:
        print(f"\nFiles/data issues reported: {len(issues)}")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nFiles/data issues reported: none")


def main() -> None:
    for input_path in (WISDM_ROOT, MOTIONSENSE_ROOT, UCI_HAR_ROOT):
        if not input_path.exists():
            raise FileNotFoundError(
                f"Configured input path does not exist: {input_path}. "
                "Edit WISDM_ROOT, MOTIONSENSE_ROOT, and UCI_HAR_ROOT at the top."
            )
    issues: list[str] = []
    source_activity_rows: dict[str, Counter[str]] = {}
    wisdm_records, _, _ = load_wisdm(
        WISDM_ROOT, issues, source_activity_rows
    )
    motionsense_records, _ = load_motionsense(
        MOTIONSENSE_ROOT, issues, source_activity_rows
    )
    uci_records, uci_info = load_uci_har(UCI_HAR_ROOT, issues)
    records = wisdm_records + motionsense_records + uci_records
    client_splits = assign_client_splits(records)
    validate_records(records, client_splits)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_RATIOS:
        split_records = [
            record for record in records if client_splits[record.client_id] == split
        ]
        write_split(OUTPUT_ROOT, split, split_records)
    for split, ratio in SPLIT_RATIOS.items():
        selected = {
            client_id
            for client_id, client_split in client_splits.items()
            if client_split == split
        }
        if selected and not math.isclose(ratio, 0.0):
            print(
                f"Split allocation {split}: {len(selected)} clients "
                f"({100 * len(selected) / len(client_splits):.1f}% of included clients)"
            )
    write_summary(
        OUTPUT_ROOT,
        records,
        client_splits,
        uci_info,
        source_activity_rows,
    )
    print_final_summary(
        records,
        client_splits,
        source_activity_rows,
        uci_info,
        issues,
    )
    print(f"\nFiles written under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
