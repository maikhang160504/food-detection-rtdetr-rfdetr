import pandas as pd
import os

dest_dir = r"d:\NCKH\Train_models\eval_results_csv"

# 1. Load per-class metrics
df_rt = pd.read_csv(os.path.join(dest_dir, "rtdetr_per_class_metrics.csv"))
df_rf = pd.read_csv(os.path.join(dest_dir, "rfdetr_per_class_metrics.csv"))

# Merge per-class
merged = pd.merge(df_rt, df_rf, on=["Class ID", "Class Name", "Instances"], suffixes=("_RTDETR", "_RFDETR"))

# Add delta and best model column
merged["Delta_mAP50_95 (RF-RT)"] = (merged["mAP@50-95_RFDETR"] - merged["mAP@50-95_RTDETR"]).round(4)
merged["Higher_mAP50_95"] = merged.apply(lambda r: "RF-DETR" if r["mAP@50-95_RFDETR"] > r["mAP@50-95_RTDETR"] else ("RT-DETR" if r["mAP@50-95_RTDETR"] > r["mAP@50-95_RFDETR"] else "Tie"), axis=1)

merged.to_csv(os.path.join(dest_dir, "model_comparison_per_class.csv"), index=False)
print("Created model_comparison_per_class.csv")

# 2. Overall Comparison CSV
overall_data = {
    "Metric": [
        "mAP@50-95",
        "mAP@50",
        "Precision",
        "Recall",
        "F1-Score",
        "Parameters (M)",
        "GFLOPs (640x640)",
        "Latency Batch=1 (ms)",
        "Throughput (FPS)"
    ],
    "RT-DETR (ResNet50)": [
        0.8746,
        0.9825,
        0.9770,
        0.9690,
        0.9730,
        32.87,
        109.32,
        39.57,
        25.3
    ],
    "RF-DETR (Medium)": [
        0.8774,
        0.9859,
        0.9756,
        0.9749,
        0.9752,
        33.60,
        99.77,
        28.97,
        34.5
    ],
    "Difference (RF - RT)": [
        "+0.0028 (+0.28%)",
        "+0.0034 (+0.34%)",
        "-0.0014 (-0.14%)",
        "+0.0059 (+0.59%)",
        "+0.0022 (+0.22%)",
        "+0.73 M (+2.2%)",
        "-9.55 GFLOPs (-8.7%)",
        "-10.60 ms (-26.8%)",
        "+9.2 FPS (+36.4%)"
    ]
}

df_overall = pd.DataFrame(overall_data)
df_overall.to_csv(os.path.join(dest_dir, "model_comparison_overall.csv"), index=False)
print("Created model_comparison_overall.csv")
