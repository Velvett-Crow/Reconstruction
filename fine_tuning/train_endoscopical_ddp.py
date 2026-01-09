'''Used to fine-tune the Depth-Anything-V2 model'''

import os, math, time, argparse, logging, pprint, random, warnings, json, csv
from pathlib import Path
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from dataset.EndoSLAM import EndoSLAM
from dataset.C3VDv2 import C3VDv2
from depth_anything_v2.dpt import DepthAnythingV2
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log

warnings.simplefilter('ignore', np.RankWarning)

# ---------------- helpers ----------------
def setup_ddp():
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(minutes=30))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return local_rank, device

def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def set_seed(seed, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def set_encoder_requires_grad(model: nn.Module, requires_grad: bool):
    n = 0
    for name, p in model.named_parameters():
        if 'pretrained' in name:
            p.requires_grad = requires_grad
            n += 1
    return n

def build_optimizer(model, base_lr):
    enc_params = [p for n,p in model.named_parameters() if ('pretrained' in n) and p.requires_grad]
    dec_params = [p for n,p in model.named_parameters() if ('pretrained' not in n) and p.requires_grad]
    return AdamW(
        [{'params': enc_params, 'lr': base_lr},
         {'params': dec_params, 'lr': base_lr * 10.0}],
        lr=base_lr, betas=(0.9, 0.999), weight_decay=0.01
    )

def cosine_lr(base_lr, it, total_it):
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(it, total_it) / float(total_it)))

def reduce_mean(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor

def add_file_logger(logger, file_path):
    fh = logging.FileHandler(file_path)
    fh.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)8s] %(message)s"))
    logger.addHandler(fh)

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser("DA-V2 ViT-B metric | 4-GPU DDP | EndoSLAM fine-tune + C3VDv2 val")
    # paths
    parser.add_argument('--pretrained', type=str,
        default="/home/jovyan/DAv2_training/depth_anything_v2_metric_hypersim_vitb.pth")
    parser.add_argument('--save-dir', type=str,
        default="/home/jovyan/DAv2_training/finetuned_vitb_endoslam_ddp")
    parser.add_argument('--train-list', type=str,
        default="/home/jovyan/Depth-Anything-V2/metric_depth/dataset/splits/EndoSLAM/train.txt")
    parser.add_argument('--val-list', type=str,
        default="/home/jovyan/Depth-Anything-V2/metric_depth/dataset/splits/EndoSLAM/val.txt")
    parser.add_argument('--c3vdv2-val', type=str,
        default="/home/jovyan/Depth-Anything-V2/metric_depth/dataset/splits/C3VDv2/val.txt")

    # model/data
    parser.add_argument('--encoder', default='vitb', choices=['vits','vitb','vitl','vitg'])
    parser.add_argument('--img-size', default=518, type=int)
    parser.add_argument('--min-depth', default=0.001, type=float)
    parser.add_argument('--max-depth', default=0.1, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--freeze-epochs', default=5, type=int)
    parser.add_argument('--bs', default=2, type=int)
    parser.add_argument('--workers', default=2, type=int)

    # optimization
    parser.add_argument('--lr', default=5e-6, type=float)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--seed', default=42, type=int)

    # logging / resume
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--status-every', type=int, default=60, help="seconds between heartbeat logs")

    args = parser.parse_args()

    # --- DDP init ---
    local_rank, device = setup_ddp()
    world_size = dist.get_world_size()
    is_main = (dist.get_rank() == 0)

    # --- folders / logging ---
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.save_dir, "logs").mkdir(parents=True, exist_ok=True)

    # Auto-resume if last.pth exists and --resume not provided
    if not args.resume:
        auto_ckpt = Path(args.save_dir, 'last.pth')
        if auto_ckpt.exists():
            args.resume = str(auto_ckpt)

    logger = init_log('global', logging.INFO)
    if logger is None:
        logger = logging.getLogger('global')
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)8s] %(message)s"))
        logger.addHandler(ch)
    if is_main:
        add_file_logger(logger, str(Path(args.save_dir, "train.log")))
        logger.info("Args:\n" + pprint.pformat(vars(args)))
        tb_writer = SummaryWriter(str(Path(args.save_dir, "logs")))
    else:
        tb_writer = None

    set_seed(args.seed, rank=dist.get_rank())
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # --- datasets / loaders ---
    size = (args.img_size, args.img_size)
    trainset = EndoSLAM(args.train_list, 'train', size=size)
    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True, drop_last=True)
    trainloader = DataLoader(
        trainset, batch_size=args.bs, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=False
    )

    # Build val loaders
    if is_main:
        valset_endo = EndoSLAM(args.val_list, 'val', size=size)
        valloader_endo = DataLoader(valset_endo, batch_size=1, shuffle=False,
                                    num_workers=1, pin_memory=True)
        valset_c3v2 = C3VDv2(args.c3vdv2_val, 'val', size=size)
        valloader_c3v2 = DataLoader(valset_c3v2, batch_size=1, shuffle=False,
                                    num_workers=1, pin_memory=True)
    else:
        valloader_endo = valloader_c3v2 = None

    # --- model ---
    model_configs = {
        'vits': {'encoder':'vits','features':64,  'out_channels':[48,96,192,384]},
        'vitb': {'encoder':'vitb','features':128, 'out_channels':[96,192,384,768]},
        'vitl': {'encoder':'vitl','features':256, 'out_channels':[256,512,1024,1024]},
        'vitg': {'encoder':'vitg','features':384, 'out_channels':[1536,1536,1536,1536]}
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})

    # Fresh start: load pretrained; Resume: let checkpoint load handle weights
    if not args.resume:
        ckpt = torch.load(args.pretrained, map_location='cpu')
        if isinstance(ckpt, dict) and 'model' in ckpt:
            ckpt = ckpt['model']
        model.load_state_dict(ckpt, strict=False)

    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        static_graph=False
    )

    criterion = SiLogLoss().to(device)

    # Freeze encoder for warmup (applies to fresh starts; resume may override below)
    n_frozen = set_encoder_requires_grad(model.module, False)
    if is_main: logger.info(f"Frozen encoder params (pretrained): {n_frozen}")

    optimizer = build_optimizer(model.module, args.lr)
    scaler = GradScaler(enabled=args.amp)

    # ---- resume support ----
    start_epoch = 1
    it_global = 0
    best_c3v2_silog = float('inf')

    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location='cpu')

        # Always load model weights from checkpoint
        model.module.load_state_dict(ck['model'], strict=False)

        # Restore optimizer before any LR modification
        try:
            optimizer.load_state_dict(ck['optimizer'])
        except Exception:
            pass

        # Restore AMP scaler if present
        if 'scaler' in ck:
            try:
                scaler.load_state_dict(ck['scaler'])
            except Exception:
                pass

        # Restore bookkeeping
        start_epoch = ck.get('epoch', 0) + 1
        it_global   = ck.get('it_global', ck.get('epoch', 0) * max(1, len(trainloader)))
        best_c3v2_silog = ck.get('best_c3vdv2_silog', float('inf'))
        if is_main:
            logger.info(f"Resuming from {args.resume} at epoch {start_epoch} (it_global={it_global})")

        # Ensure encoder is unfrozen if resuming past the freeze window
        if start_epoch > args.freeze_epochs:
            _ = set_encoder_requires_grad(model.module, True)
            # Rebuild optimizer param groups to reflect requires_grad, but **keep LR** from ckpt
            old_groups = None
            try:
                old_groups = ck['optimizer']['param_groups']
            except Exception:
                pass
            optimizer = build_optimizer(model.module, args.lr)
            if old_groups is not None:
                for gi, g in enumerate(optimizer.param_groups):
                    try:
                        g['lr'] = old_groups[gi]['lr']
                    except Exception:
                        pass

    # total iterations (used for cosine schedule); LR continuity comes from it_global
    total_iters = args.epochs * len(trainloader)

    # CSV metrics file (rank0 only)
    metrics_csv_path = Path(args.save_dir, "metrics.csv")
    if is_main and (not metrics_csv_path.exists()):
        with open(metrics_csv_path, "w", newline='') as f:
            w = csv.writer(f)
            w.writerow(["epoch","split","d1","d2","d3","abs_rel","sq_rel","rmse","rmse_log","log10","silog","train_loss","lr_enc"])

    status_json = Path(args.save_dir, "status.json")

    # -------------- training loop --------------
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_sampler.set_epoch(epoch)  # shuffles uniquely across ranks
        running = 0.0

        # Unfreeze after warmup (fresh runs only; resume may already be unfrozen)
        if (not args.resume) and (epoch == args.freeze_epochs + 1):
            _ = set_encoder_requires_grad(model.module, True)
            optimizer = build_optimizer(model.module, args.lr)
            if is_main: logger.info(f"Unfroze encoder at epoch {epoch}")

        last_heartbeat = time.time()
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad(set_to_none=True)

            img = sample['image'].to(device, non_blocking=True).float()
            depth_gt = sample['depth'].to(device, non_blocking=True).float()
            vmask = sample['valid_mask'].to(device, non_blocking=True).bool()

            # random hflip
            if random.random() < 0.5:
                img = img.flip(-1); depth_gt = depth_gt.flip(-1); vmask = vmask.flip(-1)

            with autocast(enabled=args.amp):
                pred = model(img)
                if pred.ndim == 4:
                    pred = pred[:,0]
                loss = criterion(pred, depth_gt, vmask & (depth_gt >= args.min_depth) & (depth_gt <= args.max_depth))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # cosine lr per-iter (same on all ranks) — uses restored it_global
            it_global += 1
            lr_now = cosine_lr(args.lr, it_global, total_iters)
            for gi, g in enumerate(optimizer.param_groups):
                g['lr'] = lr_now if gi == 0 else lr_now * 10.0

            # reduce loss for logging (avg across ranks)
            loss_detached = loss.detach().clone()
            loss_detached = reduce_mean(loss_detached)
            running += loss_detached.item()

            # heartbeat (rank0)
            if is_main and (time.time() - last_heartbeat) >= args.status_every:
                logger.info(f"[Epoch {epoch}/{args.epochs}] Iter {i+1}/{len(trainloader)} | LR(enc)={optimizer.param_groups[0]['lr']:.2e} | Loss(avg)={loss_detached.item():.4f}")
                with open(status_json, "w") as sj:
                    json.dump({"epoch": epoch, "iter": i+1, "iters_per_epoch": len(trainloader),
                               "lr_enc": optimizer.param_groups[0]['lr'], "loss_avg": loss_detached.item()}, sj)
                last_heartbeat = time.time()

        # epoch train loss (avg)
        train_loss = running / max(1, len(trainloader))
        if is_main and tb_writer:
            tb_writer.add_scalar('train/loss', train_loss, epoch)

        # -------- validation (rank0 only) --------
        def run_val(loader, name, mask_upper=None):
            results = {'d1':0.0,'d2':0.0,'d3':0.0,'abs_rel':0.0,'sq_rel':0.0,'rmse':0.0,'rmse_log':0.0,'log10':0.0,'silog':0.0}
            ns = 0
            model.eval()
            with torch.no_grad():
                for sample in loader:
                    img = sample['image'].to(device).float()
                    depth = sample['depth'].to(device).float()[0]
                    vmask = sample['valid_mask'].to(device).bool()[0]

                    pred = model(img)
                    if pred.ndim == 4:
                        pred = pred[:,0]
                    pred = F.interpolate(pred.unsqueeze(1), depth.shape[-2:], mode='bilinear', align_corners=True)[0,0]

                    m = (vmask == 1) & (depth >= args.min_depth)
                    m = (m & (depth <= mask_upper)) if mask_upper is not None else (m & (depth <= args.max_depth))
                    if m.sum() < 10:
                        continue

                    cur = eval_depth(pred[m], depth[m])
                    for k in results:
                        val = float(cur[k].item() if torch.is_tensor(cur[k]) else cur[k])
                        results[k] += val
                    ns += 1
            if ns == 0:
                return {k: float('nan') for k in results}
            return {k: v / ns for k, v in results.items()}

        if is_main:
            endo_res = run_val(valloader_endo, "endoslam", mask_upper=args.max_depth)
            c3v2_res = run_val(valloader_c3v2, "c3vdv2", mask_upper=args.max_depth)

            # log blocks
            def log_block(tag, res):
                logger.info(f"==== {tag} @ epoch {epoch} ====")
                logger.info(", ".join([f"{k}:{res[k]:.4f}" for k in res]))
                if tb_writer:
                    for k,v in res.items():
                        tb_writer.add_scalar(f'{tag}/{k}', v, epoch)

            log_block('val_endoslam', endo_res)
            log_block('val_c3vdv2', c3v2_res)

            # write CSV rows
            with open(metrics_csv_path, "a", newline='') as f:
                w = csv.writer(f)
                w.writerow([epoch, "endoslam", *[endo_res[k] for k in ["d1","d2","d3","abs_rel","sq_rel","rmse","rmse_log","log10","silog"]], train_loss, optimizer.param_groups[0]['lr']])
                w.writerow([epoch, "c3vdv2", *[c3v2_res[k] for k in ["d1","d2","d3","abs_rel","sq_rel","rmse","rmse_log","log10","silog"]], train_loss, optimizer.param_groups[0]['lr']])

            # checkpointing by C3VDv2 SiLog
            val_score = c3v2_res['silog']
            improved = val_score < best_c3v2_silog
            best_c3v2_silog = min(best_c3v2_silog, val_score)

            state = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_c3vdv2_silog': best_c3v2_silog,
                'args': vars(args),
                'scaler': scaler.state_dict(),
                'it_global': it_global,
                'iters_per_epoch': len(trainloader),
            }
            torch.save(state, Path(args.save_dir, 'last.pth'))
            if improved:
                tag = "best_extended.pth" if args.max_depth > 0.045 else "best.pth"
                torch.save(state, Path(args.save_dir, tag))
                logger.info(f"Saved new BEST ({tag}) at epoch {epoch} (C3VDv2 silog={val_score:.4f})")
                
                # torch.save(state, Path(args.save_dir, 'best.pth'))
                # logger.info(f"Saved new BEST at epoch {epoch} (C3VDv2 silog={val_score:.4f})")

            # end-of-epoch status
            with open(status_json, "w") as sj:
                json.dump({"epoch": epoch, "iter": len(trainloader), "iters_per_epoch": len(trainloader),
                           "lr_enc": optimizer.param_groups[0]['lr'], "loss_avg_epoch": train_loss}, sj)

            logger.info(f"Epoch {epoch} finished in {time.time()-t0:.1f}s | train_loss={train_loss:.4f}")

        # sync all ranks before next epoch
        dist.barrier()
        
        # ---- auto-stop every 10 epochs ----
        if epoch % 10 == 0:
            if is_main:
                logger.info(f"Stopping automatically after epoch {epoch} (auto-stop interval reached).")
            cleanup_ddp()
            return  # exit main() cleanly

    if is_main and tb_writer:
        tb_writer.close()
    cleanup_ddp()


if __name__ == "__main__":
    main()
