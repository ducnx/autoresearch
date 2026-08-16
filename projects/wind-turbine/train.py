"""Wind-turbine CARE benchmark entrypoint for autoresearch agents."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from energy_fault_detector import FaultDetector, generate_quickstart_config
from energy_fault_detector.evaluation.care2compare import Care2CompareDataset
from energy_fault_detector.evaluation.care_score import CAREScore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CARE To Compare wind-turbine experiment.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/CARE_To_Compare"))
    parser.add_argument("--event-id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--row-limit", type=int, default=0, help="Optional quick smoke-test row limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()

    dataset = Care2CompareDataset(path=args.data_dir)
    train_data, train_normal, test_data, test_normal = dataset.load_and_format_event_dataset(
        event_id=args.event_id,
        index_column="time_stamp",
        use_readable_columns=True,
    )
    event_info = dataset.get_event_info(args.event_id)

    if args.row_limit > 0:
        train_data = train_data.iloc[:args.row_limit]
        train_normal = train_normal.reindex(train_data.index).fillna(False)
        test_data = test_data.iloc[:args.row_limit]
        test_normal = test_normal.reindex(test_data.index).fillna(False)

    config = generate_quickstart_config(
        output_path=None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        threshold_quantile=args.threshold_quantile,
        early_stopping=False,
        validation_split=0.2,
    )
    dataset.update_c2c_config(config, wind_farm=event_info["wind_farm"], use_readable_columns=True)

    detector = FaultDetector(config=config, model_directory="workspace/models")
    metadata = detector.fit(sensor_data=train_data, normal_index=train_normal, save_models=False)
    predictions = detector.predict(sensor_data=test_data, root_cause_analysis=False)

    scorer = CAREScore()
    event_metrics = scorer.evaluate_event(
        event_start=event_info["event_start"],
        event_end=event_info["event_end"],
        event_label="anomaly",
        predicted_anomalies=predictions.predicted_anomalies,
        normal_index=test_normal,
        evaluate_until_event_end=True,
    )

    coverage = float(np.nan_to_num(event_metrics.get("f_beta_score", 0.0)))
    earliness = float(np.nan_to_num(event_metrics.get("weighted_score", 0.0)))
    max_criticality = float(event_metrics.get("max_criticality", 0.0))
    care_proxy = float(np.nanmean([coverage, earliness]))
    care_loss = 1.0 - care_proxy
    num_params_m = 0.0
    if detector.autoencoder.model is not None:
        num_params_m = detector.autoencoder.model.count_params() / 1_000_000

    elapsed = time.time() - start
    print("---")
    print(f"care_loss: {care_loss:.6f}")
    print(f"care_score: {care_proxy:.6f}")
    print(f"coverage_fbeta: {coverage:.6f}")
    print(f"earliness: {earliness:.6f}")
    print(f"max_criticality: {max_criticality:.6f}")
    print(f"val_bpb: {care_loss:.6f}")
    print(f"training_seconds: {elapsed:.3f}")
    print(f"total_seconds: {elapsed:.3f}")
    print("peak_vram_mb: 0.0")
    print("mfu_percent: 0.0")
    print(f"total_tokens_M: {len(train_data) / 1_000_000:.6f}")
    print(f"num_steps: {len(train_data)}")
    print(f"num_params_M: {num_params_m:.6f}")
    print("depth: 0")
    print(f"train_recon_error_mean: {float(np.nanmean(metadata.train_recon_error.values)):.6f}")
    if metadata.val_recon_error is not None:
        print(f"val_recon_error_mean: {float(np.nanmean(metadata.val_recon_error.values)):.6f}")


if __name__ == "__main__":
    main()
