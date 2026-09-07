"""Storage Sync Utility
Chương trình tự động đồng bộ Checkpoints, Logs, Báo cáo và Kết quả đánh giá từ Modal Volume về máy tính cá nhân.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

# Cấu hình UTF-8 cho console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_VOLUME_NAME = "food-detection-training"
DEFAULT_LOCAL_DIR = Path(__file__).resolve().parent / "synced_results"


def run_command(cmd: list) -> bool:
    """Thực thi lệnh và hiển thị output trực tiếp."""
    print(f"[*] Dang thuc thi: {' '.join(cmd)}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def sync_from_volume(
    volume_name: str = DEFAULT_VOLUME_NAME,
    target_dir: Path = DEFAULT_LOCAL_DIR,
    sync_checkpoints: bool = True,
    sync_logs: bool = True,
    sync_outputs: bool = True,
    sync_all: bool = False,
):
    print("=" * 65)
    print(f"  DONG BO DU LIEU TU MODAL VOLUME: '{volume_name}' VE MAY TINH")
    print("=" * 65)

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[+] Thu muc dich: {target_dir.resolve()}\n")

    tasks = []
    
    if sync_checkpoints or sync_all:
        tasks.append({
            "name": "Checkpoints (Trong so model best & last)",
            "remote": "/checkpoints/",
            "local": target_dir / "checkpoints",
        })

    if sync_logs or sync_all:
        tasks.append({
            "name": "Logs (Bang log 50 epochs & Bao cao so sanh)",
            "remote": "/logs/",
            "local": target_dir / "logs",
        })

    if sync_outputs or sync_all:
        tasks.append({
            "name": "Outputs (Ket qua danh gia tren tap Test)",
            "remote": "/outputs/",
            "local": target_dir / "outputs",
        })

    success_count = 0
    for task in tasks:
        print(f"\n--- [DANG TAI] {task['name']} ---")
        task["local"].mkdir(parents=True, exist_ok=True)
        
        # Gọi lệnh: modal volume get --force <volume_name> <remote_path> <local_path>
        cmd = [
            "modal",
            "volume",
            "get",
            "--force",
            volume_name,
            task["remote"],
            str(task["local"]),
        ]
        
        ok = run_command(cmd)
        if ok:
            print(f"[+] Hoan tat: {task['local'].resolve()}")
            success_count += 1
        else:
            print(f"[!] Canh bao: Khong the tai {task['remote']} (Co the thu muc chua co tren Volume neu chua chay train).")

    print("\n" + "=" * 65)
    print(f"  TONG KET: Da hoan tat {success_count}/{len(tasks)} muc.")
    print(f"  Tat ca du lieu da duoc luu tai: {target_dir.resolve()}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Dong bo ket qua tu Modal Volume ve may tinh.")
    parser.add_argument(
        "--volume",
        type=str,
        default=DEFAULT_VOLUME_NAME,
        help=f"Ten Modal Volume (mac dinh: {DEFAULT_VOLUME_NAME})",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=str(DEFAULT_LOCAL_DIR),
        help=f"Thu muc luu tru tren may (mac dinh: {DEFAULT_LOCAL_DIR})",
    )
    parser.add_argument(
        "--checkpoints-only",
        action="store_true",
        help="Chi tai thu muc checkpoints (best.pt, best.pth, last.pt, last.pth)",
    )
    parser.add_argument(
        "--logs-only",
        action="store_true",
        help="Chi tai thu muc logs va bao cao so sanh",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Tai toan bo moi du lieu tu Volume",
    )

    args = parser.parse_args()
    target_path = Path(args.dest)

    if args.checkpoints_only:
        sync_from_volume(
            volume_name=args.volume,
            target_dir=target_path,
            sync_checkpoints=True,
            sync_logs=False,
            sync_outputs=False,
        )
    elif args.logs_only:
        sync_from_volume(
            volume_name=args.volume,
            target_dir=target_path,
            sync_checkpoints=False,
            sync_logs=True,
            sync_outputs=False,
        )
    else:
        sync_from_volume(
            volume_name=args.volume,
            target_dir=target_path,
            sync_checkpoints=True,
            sync_logs=True,
            sync_outputs=True,
            sync_all=args.all,
        )


if __name__ == "__main__":
    main()
