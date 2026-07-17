"""Package predictions into the official submission zip structure:

    Task1/g_XXX.png
    Task2/g_XXX.png
    Task3/g_XXX.txt

NOTE: the competition page's text ("stored in a folder named after the team
ID") does NOT match what the actual eval server expects -- confirmed by a
real submission error, "[Errno 2] No such file or directory:
'unziped/submission/Task1'". The server unzips to a folder named after the
uploaded zip file and looks for Task1/Task2/Task3 directly inside that, with
NO team-id wrapper folder. --team-id is kept only for your own local
bookkeeping (e.g. naming the output zip), not written into the archive.

Any task directory can be omitted (partial submissions are allowed per the
competition rules -- "complete all sub-tasks or one of the sub-tasks").

Usage:
    python src/format_submission.py --team-id MyTeam \
        --task1-dir predictions/task1/validation \
        --task3-dir predictions/task3/validation \
        --out-zip submissions/MyTeam_2026-07-16.zip
"""
import argparse
import shutil
import zipfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--team-id", type=str, default=None, help="Only used for your own bookkeeping -- NOT written into the zip (see module docstring)")
    p.add_argument("--task1-dir", type=str, default=None)
    p.add_argument("--task2-dir", type=str, default=None)
    p.add_argument("--task3-dir", type=str, default=None)
    p.add_argument("--out-zip", type=str, required=True)
    args = p.parse_args()

    out_zip = Path(args.out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    staging = out_zip.parent / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    task_dirs = {"Task1": args.task1_dir, "Task2": args.task2_dir, "Task3": args.task3_dir}
    included = []
    for task_name, src_dir in task_dirs.items():
        if src_dir is None:
            continue
        src_path = Path(src_dir)
        files = sorted(src_path.glob("*"))
        if not files:
            print(f"WARNING: {task_name} source dir {src_dir} is empty, skipping")
            continue
        dst_path = staging / task_name
        shutil.copytree(src_path, dst_path)
        included.append((task_name, len(files)))

    if not included:
        raise SystemExit("No task directories provided/non-empty -- nothing to package")

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in staging.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(staging))

    shutil.rmtree(staging)

    print(f"Packaged {out_zip}:")
    for task_name, count in included:
        print(f"  {task_name}: {count} files")


if __name__ == "__main__":
    main()
