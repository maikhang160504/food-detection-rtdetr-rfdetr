import modal

app = modal.App("extract_full_tb")
vol = modal.Volume.from_name("food-detection-training")

image = modal.Image.debian_slim().pip_install("tensorboard", "pandas")

@app.function(image=image, volumes={"/vol": vol})
def get_full_per_class_history():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import glob
    import pandas as pd
    
    files = sorted(glob.glob("/vol/checkpoints/rfdetr/events*"))
    print("Found files:", files)
    
    # Extract all scalars across all event files
    all_events_by_step = {}
    for f in files:
        ea = EventAccumulator(f)
        ea.Reload()
        scalars = ea.Tags().get("scalars", [])
        for tag in scalars:
            for ev in ea.Scalars(tag):
                step = ev.step
                if step not in all_events_by_step:
                    all_events_by_step[step] = {}
                all_events_by_step[step][tag] = ev.value
    
    # Filter validation steps
    val_steps = sorted([s for s, d in all_events_by_step.items() if "val/mAP_50_95" in d])
    print(f"Total validation steps: {len(val_steps)}")
    
    rows = []
    for ep_idx, step in enumerate(val_steps, start=1):
        data = all_events_by_step[step]
        row = {"epoch": ep_idx, "step": step}
        row.update(data)
        rows.append(row)
        
    df = pd.DataFrame(rows)
    print("Columns extracted:", df.columns.tolist())
    print(f"Shape: {df.shape}")
    
    # Save full raw per-class csv
    df.to_csv("/vol/checkpoints/rfdetr/metrics_full_per_class.csv", index=False)
    vol.commit()
    return df.to_dict(orient="records")

@app.local_entrypoint()
def main():
    res = get_full_per_class_history.remote()
    print(f"Extracted {len(res)} epochs with full per-class columns!")
