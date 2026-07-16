"""Package predictions into the official submission zip structure:

    <team_id>/
      Task1/g_XXX.png
      Task2/g_XXX.png
      Task3/g_XXX.txt

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
    p.add_argument("--team-id", type=str, required=True)
    p.add_argument("--task1-dir", type=str, default=None)
    p.add_argument("--task2-dir", type=str, default=None)
    p.add_argument("--task3-dir", type=str, default=None)
    p.add_argument("--out-zip", type=str, required=True)
    args = p.parse_args()

    out_zip = Path(args.out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    staging = out_zip.parent / f"_staging_{args.team_id}"
    if staging.exists():
        shutil.rmtree(staging)
    team_dir = staging / args.team_id
    team_dir.mkdir(parents=True)

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
        dst_path = team_dir / task_name
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
