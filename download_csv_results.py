import os
import modal

def download():
    vol = modal.Volume.from_name('food-detection-training')
    
    # Destination directory
    dest_dir = r"d:\NCKH\Train_models\eval_results_csv"
    os.makedirs(dest_dir, exist_ok=True)
    
    files_to_download = [
        # RT-DETR
        ("outputs/rtdetr_eval/per_class_metrics.csv", "rtdetr_per_class_metrics.csv"),
        ("outputs/rtdetr_eval/confusion_matrix.csv", "rtdetr_confusion_matrix.csv"),
        ("outputs/rtdetr_eval/confusion_matrix_normalized.csv", "rtdetr_confusion_matrix_normalized.csv"),
        ("outputs/rtdetr_eval/rt-detr_test_report.md", "rtdetr_test_report.md"),
        ("outputs/rtdetr_eval/rt-detr_test_report.json", "rtdetr_test_report.json"),
        
        # RF-DETR
        ("outputs/rfdetr_eval/per_class_metrics.csv", "rfdetr_per_class_metrics.csv"),
        ("outputs/rfdetr_eval/confusion_matrix.csv", "rfdetr_confusion_matrix.csv"),
        ("outputs/rfdetr_eval/confusion_matrix_normalized.csv", "rfdetr_confusion_matrix_normalized.csv"),
        ("outputs/rfdetr_eval/rf-detr_test_report.md", "rfdetr_test_report.md"),
        ("outputs/rfdetr_eval/rf-detr_test_report.json", "rfdetr_test_report.json"),
        
        # Comparison report
        ("logs/comparison_report.md", "comparison_report.md"),
    ]
    
    for remote_path, local_name in files_to_download:
        local_path = os.path.join(dest_dir, local_name)
        try:
            print(f"Downloading {remote_path} -> {local_path} ...")
            data = b"".join(vol.read_file(remote_path))
            with open(local_path, "wb") as f:
                f.write(data)
            print(f"Successfully saved {local_name} ({len(data)} bytes)")
        except Exception as e:
            print(f"Error downloading {remote_path}: {e}")

if __name__ == "__main__":
    download()
