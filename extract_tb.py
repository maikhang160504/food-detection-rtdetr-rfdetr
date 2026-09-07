import modal

app = modal.App("extract_tb")
vol = modal.Volume.from_name("food-detection-training")

image = modal.Image.debian_slim().pip_install("tensorboard", "pandas")

@app.function(image=image, volumes={"/vol": vol})
def get_tb_history():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import glob
    import pandas as pd
    
    files = sorted(glob.glob("/vol/checkpoints/rfdetr/events*"))
    print("Found files:", files)
    
    all_metrics = []
    for f in files:
        ea = EventAccumulator(f)
        ea.Reload()
        scalars = ea.Tags().get("scalars", [])
        if "val/mAP_50_95" in scalars:
            map50_95_events = ea.Scalars("val/mAP_50_95")
            map50_events = {e.step: e.value for e in ea.Scalars("val/mAP_50")} if "val/mAP_50" in scalars else {}
            prec_events = {e.step: e.value for e in ea.Scalars("val/precision")} if "val/precision" in scalars else {}
            rec_events = {e.step: e.value for e in ea.Scalars("val/recall")} if "val/recall" in scalars else {}
            loss_events = {e.step: e.value for e in ea.Scalars("val/loss")} if "val/loss" in scalars else {}
            loss_bbox = {e.step: e.value for e in ea.Scalars("val/loss_bbox")} if "val/loss_bbox" in scalars else {}
            loss_ce = {e.step: e.value for e in ea.Scalars("val/loss_ce")} if "val/loss_ce" in scalars else {}
            loss_giou = {e.step: e.value for e in ea.Scalars("val/loss_giou")} if "val/loss_giou" in scalars else {}
            train_loss = {e.step: e.value for e in ea.Scalars("train/loss")} if "train/loss" in scalars else {}
            lr_events = {e.step: e.value for e in ea.Scalars("train/lr")} if "train/lr" in scalars else {}

            for e in map50_95_events:
                step = e.step
                all_metrics.append({
                    "step": step,
                    "val_map50_95": e.value,
                    "val_map50": map50_events.get(step, 0),
                    "val_precision": prec_events.get(step, 0),
                    "val_recall": rec_events.get(step, 0),
                    "val_loss": loss_events.get(step, 0),
                    "val_loss_bbox": loss_bbox.get(step, 0),
                    "val_loss_ce": loss_ce.get(step, 0),
                    "val_loss_giou": loss_giou.get(step, 0),
                    "train_loss": train_loss.get(step, 0),
                    "lr": lr_events.get(step, 1e-4),
                })
    
    df = pd.DataFrame(all_metrics)
    df = df.drop_duplicates(subset=["step"]).sort_values(by="step").reset_index(drop=True)
    df["epoch"] = range(1, len(df) + 1)
    print(df.to_string())
    df.to_csv("/vol/checkpoints/rfdetr/full_46_epochs_metrics.csv", index=False)
    vol.commit()
    return df.to_dict(orient="records")

@app.local_entrypoint()
def main():
    res = get_tb_history.remote()
    print(f"Total epochs extracted: {len(res)}")
