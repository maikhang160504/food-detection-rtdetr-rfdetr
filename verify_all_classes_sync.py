import pandas as pd
import numpy as np

# Paths
dir_csv = r"d:\NCKH\Train_models\eval_results_csv"

# Load files
rt_metrics = pd.read_csv(f"{dir_csv}/rtdetr_per_class_metrics.csv")
rt_cm_norm = pd.read_csv(f"{dir_csv}/rtdetr_confusion_matrix_normalized.csv", index_col=0)
rt_cm_raw = pd.read_csv(f"{dir_csv}/rtdetr_confusion_matrix.csv", index_col=0)

rf_metrics = pd.read_csv(f"{dir_csv}/rfdetr_per_class_metrics.csv")
rf_cm_norm = pd.read_csv(f"{dir_csv}/rfdetr_confusion_matrix_normalized.csv", index_col=0)
rf_cm_raw = pd.read_csv(f"{dir_csv}/rfdetr_confusion_matrix.csv", index_col=0)

print(f"Total RT classes: {len(rt_metrics)}, Total RF classes: {len(rf_metrics)}")
print(f"Total RT instances: {rt_metrics['Instances'].sum()}, Total RF instances: {rf_metrics['Instances'].sum()}")

# Check alignment of Class Names and Instances
instances_match = (rt_metrics['Instances'] == rf_metrics['Instances']).all()
names_match = (rt_metrics['Class Name'] == rf_metrics['Class Name']).all()
print(f"Class Names match 100%: {names_match}")
print(f"Instances match 100%: {instances_match}")

print("\n" + "="*80)
print(f"{'Class':<15} | {'Inst':<5} | {'RT Rec':<7} {'RT CM_Diag':<10} {'RT TP/Tot':<10} | {'RF Rec':<7} {'RF CM_Diag':<10} {'RF TP/Tot':<10}")
print("="*80)

rt_diffs = []
rf_diffs = []

for idx, row in rt_metrics.iterrows():
    c_name = row['Class Name']
    inst = row['Instances']
    
    # RT
    rt_rec = row['Recall']
    rt_diag = rt_cm_norm.loc[c_name, c_name]
    rt_tp = rt_cm_raw.loc[c_name, c_name]
    rt_tot = rt_cm_raw.loc[c_name].sum()  # true ground truths
    rt_bg = rt_cm_raw.loc[c_name, 'background']
    rt_diff = abs(rt_rec - rt_diag)
    rt_diffs.append(rt_diff)
    
    # RF
    rf_row = rf_metrics[rf_metrics['Class Name'] == c_name].iloc[0]
    rf_rec = rf_row['Recall']
    rf_diag = rf_cm_norm.loc[c_name, c_name]
    rf_tp = rf_cm_raw.loc[c_name, c_name]
    rf_tot = rf_cm_raw.loc[c_name].sum()
    rf_bg = rf_cm_raw.loc[c_name, 'background']
    rf_diff = abs(rf_rec - rf_diag)
    rf_diffs.append(rf_diff)
    
    print(f"{c_name:<15} | {inst:<5} | {rt_rec:<7.4f} {rt_diag:<10.4f} {rt_tp}/{rt_tot} (miss {rt_bg}) | {rf_rec:<7.4f} {rf_diag:<10.4f} {rf_tp}/{rf_tot} (miss {rf_bg})")

print("="*80)
print(f"RT-DETR Max difference between Table Recall & CM Diagonal: {max(rt_diffs):.4f}")
print(f"RF-DETR Max difference between Table Recall & CM Diagonal: {max(rf_diffs):.4f}")
