import pandas as pd
from pathlib import Path

old_dir, new_dir, out_dir = Path("output"), Path("output_20k"), Path("output_merged")
out_dir.mkdir(exist_ok=True)

# 1. Overlap check at 10000 threads: the new run must reproduce the old one
key = ["model", "thread_count", "quantization"]
old_s = pd.read_csv(old_dir / "dataset_evaluation_summary.csv")
new_s = pd.read_csv(new_dir / "dataset_evaluation_summary.csv")
chk = old_s.merge(new_s, on=key, suffixes=("_old", "_new"))
print("Rows compared:", len(chk))
print("Max PSNR difference:", (chk.mean_psnr_db_old - chk.mean_psnr_db_new).abs().max())
print("Max recovered_objects difference:", (chk.recovered_objects_old - chk.recovered_objects_new).abs().max())

# 2. Append only the new thread counts, keep old rows untouched
names = ["dataset_evaluation_summary.csv", "yolo_benchmark_results.csv"] + [
    p.name for p in old_dir.glob("*_results.csv") if p.name != "yolo_benchmark_results.csv"
]
for name in names:
    old = pd.read_csv(old_dir / name)
    new = pd.read_csv(new_dir / name)
    merged = pd.concat([old, new[new["thread_count"] > old["thread_count"].max()]], ignore_index=True)
    if name == "dataset_evaluation_summary.csv":
        merged = merged.sort_values(["model", "thread_count"], kind="stable")
    merged.to_csv(out_dir / name, index=False)
    print(f"{name}: {len(old):,} + {len(merged) - len(old):,} = {len(merged):,} rows")