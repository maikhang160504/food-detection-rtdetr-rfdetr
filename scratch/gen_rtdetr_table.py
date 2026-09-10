import pandas as pd

df = pd.read_csv('synced_results_v4/logs/rtdetr/rtdetr_epoch_logs.csv')
print(f'Total epochs: {len(df)}')

lines = []
lines.append('| Epoch | Train Loss | Class Loss | GIoU Loss | Learning Rate | Val Precision | Val Recall | Val mAP@50 | Val mAP@50-95 | Time (s) |')
lines.append('| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |')

for _, r in df.iterrows():
    ep = int(r['Epoch'])
    tl = f"{r['Train loss']:.4f}"
    cl = f"{r['Class loss']:.4f}"
    gl = f"{r['GIoU']:.4f}"
    lr = f"{float(r['Learning rate']):.3e}"
    vp = f"{float(r['Val Precision']):.4f}" if pd.notna(r.get('Val Precision')) else '-'
    vr = f"{float(r['Val Recall']):.4f}" if pd.notna(r.get('Val Recall')) else '-'
    m50 = f"{float(r['val_mAP50']):.4f}" if pd.notna(r.get('val_mAP50')) else '-'
    m50_95 = f"{float(r['val_mAP50_95']):.4f}" if pd.notna(r.get('val_mAP50_95')) else '-'
    t = f"{r['Epoch Time (s)']:.1f}s"
    
    if ep in [58, 59]:
        lines.append(f'| **{ep}** | **{tl}** | **{cl}** | **{gl}** | **{lr}** | **{vp}** | **{vr}** | **{m50}** | **{m50_95}** ⭐ | **{t}** |')
    else:
        lines.append(f'| {ep} | {tl} | {cl} | {gl} | {lr} | {vp} | {vr} | {m50} | {m50_95} | {t} |')

with open('scratch/full_rtdetr_table.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print('Saved to scratch/full_rtdetr_table.md')
