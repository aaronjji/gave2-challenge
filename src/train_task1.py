"""Task 1 (CFP-only AV segmentation) training entrypoint.

Usage:
    python src/train_task1.py --fold 0 --epochs 60 --data-root data/raw/GAVE2_preliminary
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets.gave_dataset import GaveAVDataset  # noqa: E402
from datasets.splits import kfold_case_ids  # noqa: E402
from losses.rrloss import BCE3Loss, RRLoss  # noqa: E402
from models.rrwnet import build_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data/raw/GAVE2_preliminary")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--steps-per-epoch", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=True)
    p.add_argument("--out-dir", type=str, default="runs/task1")
    p.add_argument("--max-steps", type=int, default=None, help="Smoke-test override: stop after N total steps")
    p.add_argument("--max-seconds", type=float, default=None, help="Wall-clock budget (e.g. Kaggle's ~9-12h session cap); checkpoints and exits cleanly before the limit")
    p.add_argument("--checkpoint-every-epochs", type=int, default=2)
    p.add_argument("--resume", action="store_true", help="Resume from out_dir/fold{N}/latest.pth if present")
    p.add_argument("--seed", type=int, default=77)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    splits = kfold_case_ids(n_cases=50, n_folds=args.n_folds, seed=args.seed)
    train_ids, val_ids = splits[args.fold]
    print(f"Fold {args.fold}: {len(train_ids)} train / {len(val_ids)} val cases")

    train_ds = GaveAVDataset(
        args.data_root, split="training", case_ids=train_ids, patch_size=args.patch_size,
        use_ffa=False, train=True, seed=args.seed,
    )
    val_ds = GaveAVDataset(
        args.data_root, split="training", case_ids=val_ids, patch_size=args.patch_size,
        use_ffa=False, train=False, seed=args.seed + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("task1", base_ch=args.base_ch, iterations=args.iterations, pretrained=args.pretrained).to(device)

    criterion = RRLoss(BCE3Loss())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    out_dir = Path(args.out_dir) / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    global_step = 0
    start_epoch = 0
    latest_path = out_dir / "latest.pth"
    log_path = out_dir / "train_log.csv"

    if args.resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        print(f"Resumed from {latest_path}: epoch={start_epoch} global_step={global_step}")
    else:
        with open(log_path, "w") as f:
            f.write("epoch,step,loss,elapsed_s\n")

    def save_checkpoint(path: Path, epoch: int):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            },
            path,
        )

    t_start = time.time()
    last_epoch_duration = 0.0
    for epoch in range(start_epoch, args.epochs):
        if args.max_seconds is not None:
            elapsed_so_far = time.time() - t_start
            # Bail out before starting an epoch we likely can't finish within budget.
            if elapsed_so_far + last_epoch_duration * 1.2 > args.max_seconds:
                print(f"[max_seconds budget] stopping before epoch {epoch+1} (elapsed={elapsed_so_far:.0f}s, budget={args.max_seconds:.0f}s)")
                save_checkpoint(latest_path, epoch)
                return
        epoch_t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        train_iter = iter(train_loader)
        for step in range(args.steps_per_epoch):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            roi = batch["roi"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                predictions = model(image)
                loss = criterion(predictions, label, roi)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if args.max_steps is not None and global_step >= args.max_steps:
                print(f"[smoke test] reached max_steps={args.max_steps}, stopping")
                torch.save(model.state_dict(), out_dir / "smoke_test_ckpt.pth")
                return

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t_start
        last_epoch_duration = time.time() - epoch_t0
        print(f"epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  elapsed={elapsed:.0f}s  epoch_dur={last_epoch_duration:.0f}s")
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{global_step},{avg_loss:.6f},{elapsed:.1f}\n")

        if (epoch + 1) % args.checkpoint_every_epochs == 0 or epoch == args.epochs - 1:
            save_checkpoint(latest_path, epoch + 1)
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            torch.save(model.state_dict(), out_dir / f"epoch{epoch+1}.pth")

    torch.save(model.state_dict(), out_dir / "final.pth")


if __name__ == "__main__":
    main()
