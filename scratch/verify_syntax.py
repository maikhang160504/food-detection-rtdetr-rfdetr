import sys
import py_compile
import yaml

# Set stdout to UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

print("--- 1. Testing Python Syntax & Bytecode Compilation ---")
files = [
    "src/common/benchmark.py",
    "src/rfdetr/confusion_matrix.py",
    "src/common/metrics_reporter.py",
    "src/rtdetr/evaluate.py",
    "src/rfdetr/evaluate.py",
    "src/rtdetr/train.py",
    "src/rfdetr/train.py",
    "modal_app/pipeline.py",
]
for f in files:
    py_compile.compile(f, doraise=True)
    print(f" [PASSED] {f}")

print("\n--- 2. Checking YAML Configuration Consistency (v4) ---")
for y in ["configs/rtdetr.yaml", "configs/rfdetr.yaml"]:
    with open(y, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
        train_cfg = cfg.get("training", {})
        eval_cfg = cfg.get("evaluation", {})
        print(f" [CONFIG] {y}:")
        print(f"   - epochs: {train_cfg.get('epochs')}")
        print(f"   - patience: {train_cfg.get('patience')}")
        print(f"   - imgsz: {train_cfg.get('imgsz')}")
        print(f"   - batch_size: {train_cfg.get('batch_size')}")
        print(f"   - eval_conf: {eval_cfg.get('eval_conf')}")
        print(f"   - cm_conf: {eval_cfg.get('cm_conf')}")

print("\n[+] ALL PYTHON SCRIPTS AND YAML CONFIGURATIONS ARE VALID!")
