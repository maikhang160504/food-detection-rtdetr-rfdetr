"""Metrics Reporter Module
Tính toán và xuất báo cáo độ chính xác:
- Precision, Recall, mAP50, mAP50-95
- Độ chính xác của từng lớp trên tập Test
- Độ chính xác tổng thể trên toàn bộ tập dữ liệu
- Bảng so sánh tổng hợp giữa các mô hình
"""
import os
import json
import pandas as pd
from typing import Dict, List, Any


class MetricsReporter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_test_report(
        self,
        model_name: str,
        overall_metrics: Dict[str, float],
        per_class_metrics: List[Dict[str, Any]],
    ) -> str:
        report_data = {
            "model": model_name,
            "overall": overall_metrics,
            "per_class": per_class_metrics,
        }

        # 1. Save JSON
        json_path = os.path.join(self.output_dir, f"{model_name.lower()}_test_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # 2. Build Markdown Table
        df_class = pd.DataFrame(per_class_metrics)
        md_content = f"# Báo Cáo Đánh Giá Mô Hình: {model_name}\n\n"
        md_content += "## 1. Độ Chính Xác Tổng Thể Trên Tập Test\n\n"
        md_content += f"- **Precision (P)**: {overall_metrics.get('precision', 0.0):.4f}\n"
        md_content += f"- **Recall (R)**: {overall_metrics.get('recall', 0.0):.4f}\n"
        md_content += f"- **mAP@50**: {overall_metrics.get('map50', 0.0):.4f}\n"
        md_content += f"- **mAP@50-95**: {overall_metrics.get('map50_95', 0.0):.4f}\n\n"

        md_content += "## 2. Độ Chính Xác Từng Lớp Trên Tập Test (Per-Class Accuracy)\n\n"
        if not df_class.empty:
            md_content += df_class.to_markdown(index=False) + "\n\n"

        md_path = os.path.join(self.output_dir, f"{model_name.lower()}_test_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"[+] Saved evaluation report to: {md_path}")
        return md_path

    @staticmethod
    def generate_comparison_report(
        rtdetr_report_json: str,
        rfdetr_report_json: str,
        output_file: str,
    ) -> str:
        with open(rtdetr_report_json, "r", encoding="utf-8") as f:
            rt_data = json.load(f)
        with open(rfdetr_report_json, "r", encoding="utf-8") as f:
            rf_data = json.load(f)

        rows = [
            {
                "Tiêu chí": "mAP@50-95 (Test)",
                "RT-DETR": f"{rt_data['overall'].get('map50_95', 0.0):.4f}",
                "RF-DETR": f"{rf_data['overall'].get('map50_95', 0.0):.4f}",
            },
            {
                "Tiêu chí": "mAP@50 (Test)",
                "RT-DETR": f"{rt_data['overall'].get('map50', 0.0):.4f}",
                "RF-DETR": f"{rf_data['overall'].get('map50', 0.0):.4f}",
            },
            {
                "Tiêu chí": "Precision (Test)",
                "RT-DETR": f"{rt_data['overall'].get('precision', 0.0):.4f}",
                "RF-DETR": f"{rf_data['overall'].get('precision', 0.0):.4f}",
            },
            {
                "Tiêu chí": "Recall (Test)",
                "RT-DETR": f"{rt_data['overall'].get('recall', 0.0):.4f}",
                "RF-DETR": f"{rf_data['overall'].get('recall', 0.0):.4f}",
            },
        ]

        df_comp = pd.DataFrame(rows)
        md = "# Báo Cáo So Sánh RT-DETR vs RF-DETR\n\n"
        md += df_comp.to_markdown(index=False)
        md += "\n\n"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md)

        return output_file
