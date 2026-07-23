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
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets.gave_dataset import GaveAVDataset  # noqa: E402
from datasets.pseudo_label_dataset import PseudoLabelDataset  # noqa: E402
from datasets.splits import kfold_case_ids  # noqa: E402
from losses.rrloss import BCE3Loss, RRLoss, RRClDiceLoss  # noqa: E402
from losses.cldice import ArteryVeinClDiceLoss  # noqa: E402
from models.rrwnet import build_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data/raw/GAVE2_preliminary")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument(
        "--patch-size", type=int, default=384,
        help="384 verified stable+fast on the T550 (bf16, 3.09GB peak, ~9s/step); "
             "448+ triggers a severe slowdown from Windows shared-GPU-memory fallback near the 4GB ceiling",
    )
    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--steps-per-epoch", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--amp-dtype", type=str, default="bf16", choices=["none", "fp16", "bf16"],
        help="fp16 autocast produced NaN forward passes on the T550 (verified empirically) -- bf16 avoids the overflow (same exponent range as fp32) at a similar memory cost; 'none' disables mixed precision entirely",
    )
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=True)
    p.add_argument(
        "--pos-weight", type=float, default=5.0,
        help="Upweights vessel-positive pixels in the BCE loss. Real leaderboard data (2026-07-17) showed our "
             "Sensitivity trails the #1 team badly (~0.6-0.75 vs ~0.96-0.97) while our DSC is actually higher -- "
             "we're too conservative, not too imprecise. 0 disables (matches the original baseline's unweighted loss).",
    )
    p.add_argument(
        "--cldice-weight", type=float, default=0.3,
        help="Weight for the soft-clDice topology loss on the final prediction (0 disables). Targets the COR/INF "
             "gap directly -- real data showed COR ~0.1-0.3 vs the #1 team's ~0.78, INF ~0.7-0.9 vs their ~0.22.",
    )
    p.add_argument(
        "--vein-pos-weight", type=float, default=None,
        help="Separate (higher) pos_weight for the vein channel only, mirroring train_task2.py. Real leaderboard "
             "data (2026-07-19) showed Task1's vein topology can crater (COR 0.54->0.32, INF 0.45->0.68) from further "
             "training under symmetric weighting while artery improves -- None (default) keeps it equal to --pos-weight.",
    )
    p.add_argument(
        "--vein-topology-ratio", type=float, default=1.0,
        help="Relative weight of vein vs artery inside the clDice term (artery fixed at 1.0). Same motivation as "
             "--vein-pos-weight. Task2's leaderboard data (2026-07-19) showed an aggressive ratio (2.0) overcorrects "
             "and drags the OTHER channel's topology down more than it helps -- keep this mild (~1.2-1.4) for Task1.",
    )
    p.add_argument("--out-dir", type=str, default="runs/task1")
    p.add_argument("--max-steps", type=int, default=None, help="Smoke-test override: stop after N total steps")
    p.add_argument("--max-seconds", type=float, default=None, help="Wall-clock budget (e.g. Kaggle's ~9-12h session cap); checkpoints and exits cleanly before the limit")
    p.add_argument("--checkpoint-every-epochs", type=int, default=2)
    p.add_argument(
        "--val-every-epochs", type=int, default=5,
        help="Compute held-out validation loss every N epochs and save best.pth on improvement. "
             "Previously val_loader was created but never used -- training was blind to overfitting.",
    )
    p.add_argument("--resume", action="store_true", help="Resume from out_dir/fold{N}/latest.pth if present")
    p.add_argument("--seed", type=int, default=77)
    p.add_argument("--device", type=str, default=None, help="Override device (e.g. 'cpu') -- for smoke-testing without touching a GPU that's busy elsewhere")
    p.add_argument(
        "--pseudo-images-dir", type=str, default=None,
        help="Self-training: dir of unlabeled images to pseudo-label from --pseudo-pred-dir "
             "(e.g. data/raw/GAVE2_preliminary/validation/images). None disables self-training entirely.",
    )
    p.add_argument("--pseudo-masks-dir", type=str, default=None, help="ROI masks for the pseudo images")
    p.add_argument(
        "--pseudo-pred-dir", type=str, default=None,
        help="Our own ensemble's quantized probability-map predictions on the pseudo images "
             "(e.g. predictions/task1/validation_ensemble) -- source of the pseudo-labels",
    )
    p.add_argument(
        "--pseudo-weight", type=float, default=0.3,
        help="Fraction of training samples drawn from the pseudo-labeled set per epoch (0-1). "
             "Kept well under 0.5 -- pseudo-labels are noisier than real GT even after confidence filtering.",
    )
    p.add_argument(
        "--pseudo-case-limit", type=int, default=None,
        help="Use only the first N pseudo cases (sorted) -- for a cheap initial validated trial before scaling to the full unlabeled set.",
    )
    return p.parse_args()


@torch.no_grad()
def run_validation(model, val_loader, criterion, device, amp_enabled, amp_torch_dtype):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in val_loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        roi = batch["roi"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_torch_dtype, enabled=amp_enabled):
            predictions = model(image)
            loss = criterion(predictions, label, roi)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


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

    if args.pseudo_images_dir:
        assert args.pseudo_masks_dir and args.pseudo_pred_dir, "--pseudo-masks-dir and --pseudo-pred-dir required with --pseudo-images-dir"
        pseudo_names = sorted(p.stem for p in Path(args.pseudo_images_dir).glob("*.png"))
        if args.pseudo_case_limit:
            pseudo_names = pseudo_names[: args.pseudo_case_limit]
        pseudo_ds = PseudoLabelDataset(
            images_dir=args.pseudo_images_dir, masks_dir=args.pseudo_masks_dir, pred_dir=args.pseudo_pred_dir,
            case_names=pseudo_names, patch_size=args.patch_size, use_ffa=False, seed=args.seed,
        )
        print(f"Self-training: mixing in {len(pseudo_names)} pseudo-labeled cases at weight={args.pseudo_weight}")
        combined_ds = ConcatDataset([train_ds, pseudo_ds])
        real_w = (1.0 - args.pseudo_weight) / len(train_ds)
        pseudo_w = args.pseudo_weight / len(pseudo_ds)
        sample_weights = [real_w] * len(train_ds) + [pseudo_w] * len(pseudo_ds)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(combined_ds), replacement=True)
        train_loader = DataLoader(combined_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("task1", base_ch=args.base_ch, iterations=args.iterations, pretrained=args.pretrained).to(device)

    base_criterion = BCE3Loss(
        pos_weight=args.pos_weight if args.pos_weight > 0 else None,
        vein_pos_weight=args.vein_pos_weight if args.pos_weight > 0 else None,
    )
    if args.cldice_weight > 0:
        cldice_loss = ArteryVeinClDiceLoss(artery_weight=1.0, vein_weight=args.vein_topology_ratio)
        criterion = RRClDiceLoss(base_criterion, cldice_loss, cldice_weight=args.cldice_weight)
    else:
        criterion = RRLoss(base_criterion)
    criterion = criterion.to(device)  # pos_weight is a buffer, must move with the module
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    amp_enabled = args.amp_dtype != "none" and device.type == "cuda"
    amp_torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "none": None}[args.amp_dtype]
    # GradScaler is only needed (and only valid) for fp16 -- bf16's exponent
    # range matches fp32 so it doesn't need loss scaling.
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and args.amp_dtype == "fp16")

    out_dir = Path(args.out_dir) / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")
    latest_path = out_dir / "latest.pth"
    best_path = out_dir / "best.pth"
    log_path = out_dir / "train_log.csv"

    if args.resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {latest_path}: epoch={start_epoch} global_step={global_step}")
    else:
        with open(log_path, "w") as f:
            f.write("epoch,step,loss,val_loss,elapsed_s\n")

    def save_checkpoint(path: Path, epoch: int):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
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
            with torch.amp.autocast("cuda", dtype=amp_torch_dtype, enabled=amp_enabled):
                predictions = model(image)
                loss = criterion(predictions, label, roi)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if torch.isnan(loss):
                raise RuntimeError(
                    f"NaN loss at step {global_step} (epoch {epoch+1}) -- amp_dtype={args.amp_dtype}. "
                    "If this is fp16, switch to --amp-dtype bf16 or none; the model's forward pass is "
                    "numerically unstable under fp16 autocast on some GPUs (verified on this project's T550)."
                )

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

        val_loss_str = ""
        val_loss = None
        if args.val_every_epochs > 0 and ((epoch + 1) % args.val_every_epochs == 0 or epoch == args.epochs - 1):
            val_loss = run_validation(model, val_loader, criterion, device, amp_enabled, amp_torch_dtype)
            val_loss_str = f"  val_loss={val_loss:.4f}"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(best_path, epoch + 1)
                val_loss_str += " (best, saved)"

        print(f"epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}{val_loss_str}  elapsed={elapsed:.0f}s  epoch_dur={last_epoch_duration:.0f}s")
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{global_step},{avg_loss:.6f},{val_loss if val_loss is not None else ''},{elapsed:.1f}\n")

        if (epoch + 1) % args.checkpoint_every_epochs == 0 or epoch == args.epochs - 1:
            save_checkpoint(latest_path, epoch + 1)
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            torch.save(model.state_dict(), out_dir / f"epoch{epoch+1}.pth")

    torch.save(model.state_dict(), out_dir / "final.pth")


if __name__ == "__main__":
    main()
