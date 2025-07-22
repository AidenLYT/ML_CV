import os
import os.path as osp
import random

import numpy as np
import pandas as pd
import argparse
from datetime import datetime

from torch.cuda.amp import GradScaler, autocast

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

from datasets import Proj_Dataset
from models import *


IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD = [0.229, 0.224, 0.225]


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--lr', type=float,default='4e-3')
    argparser.add_argument('--optim_type', default='adam')
    argparser.add_argument('--arch_ver', default='ver1')
    argparser.add_argument('--freeze', action='store_true')
    argparser.add_argument('--ver_name', default="")
    argparser.add_argument('--epochs', default=25)
    argparser.add_argument('--batch', default=32)
    argparser.add_argument('--mixup_alpha', default=0.2)
    argparser.add_argument('--label_smooth', default=0.1)
    argparser.add_argument('--freeze_epochs', default=5)
    args = argparser.parse_args()
    return args


def split_trainval(num_train=45, num_val=10):
    trainval_annos = pd.read_csv('datasets/train_anno.csv')

    categories = sorted(trainval_annos['cls'].unique())
    train_annos, val_annos = [], []
    for c in categories:
        idxs = np.arange(num_train + num_val)
        np.random.shuffle(idxs)
        tgt_df = trainval_annos.groupby('cls').get_group(c).reset_index(drop=True)
        train_annos.append(tgt_df.loc[idxs[:num_train]])
        val_annos.append(tgt_df.loc[idxs[num_train:]])

    train_annos = pd.concat(train_annos).reset_index(drop=True)
    val_annos = pd.concat(val_annos).reset_index(drop=True)

    return train_annos, val_annos


def run_val_epoch(model, val_loader, device, criterion):
    model.eval()

    sum_loss = 0
    correct = 0
    num_samples = 0

    with torch.no_grad(), autocast():
        for idx, (img, gt_y) in enumerate(val_loader):
            img, gt_y = img.to(device), gt_y.to(device)

            num_batch = gt_y.shape[0]
            img = img.to(device).float()
            gt_y = gt_y.to(device).long()
            out = model(img)
            _, pred = torch.max(out, 1)
            correct += pred.eq(gt_y.data).sum().item()

            loss = criterion(out, gt_y)
            sum_loss += num_batch * (loss.item())
            num_samples += num_batch

    loss = sum_loss / num_samples
    acc = 100 * correct / num_samples

    return loss, acc


def rand_mixup(x, y, alpha):
    if alpha <= 0:  return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    return lam*x + (1-lam)*x[idx], lam*y + (1-lam)*y[idx], lam


def run_trainval(model, tr_dl, va_dl, args, device, opt, sched, criterion, scaler, writer):
    best_acc = 0.0
    backbone_params = list(model.backbone.parameters())
    for epoch in range(args.epochs):
        start_time = datetime.now()
        model.train()
        ep_loss = 0.0
        ep_pred_y, ep_gt_y = [], []
        if epoch == args.freeze_epochs:
            for p in backbone_params:
                p.requires_grad = True
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            yb_onehot = torch.zeros(len(yb), 50, device=device).scatter_(1, yb[:, None], 1.0)
            xb, yb_mix, _ = rand_mixup(xb, yb_onehot, args.mixup_alpha)
            opt.zero_grad(set_to_none=True)
            with autocast():
                logits = model(xb)
                loss = criterion(logits, yb_mix)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ep_loss += loss.item() * yb.size(0)
            ep_pred_y.append(logits.detach().argmax(dim=1).cpu())
            ep_gt_y.append(yb.cpu())

        end_time = datetime.now()
        print(f"Time elapsed {end_time - start_time}")

        ep_pred_y = torch.cat(ep_pred_y)
        ep_gt_y = torch.cat(ep_gt_y)
        train_loss = ep_loss / len(ep_gt_y)
        train_acc = 100 * (ep_gt_y == ep_pred_y).to(float).mean().item()
        val_loss, val_acc = run_val_epoch(model, va_dl, device, criterion)

        print(f"[train-{epoch + 1}/{args.epochs}] loss: {train_loss:.6f} | acc: {train_acc:.3f}%")
        print(f"[val-{epoch + 1}/{args.epochs}] loss: {val_loss:.6f} | acc: {val_acc:.3f}%")
        writer.add_scalar('ep_loss/train', train_loss, epoch + 1)
        writer.add_scalar('ep_loss/val', val_loss, epoch + 1)
        writer.add_scalar('ep_acc/train', train_acc, epoch + 1)
        writer.add_scalar('ep_acc/val', val_acc, epoch + 1)

        torch.save(model.state_dict(), osp.join(args.output, "last.pt"))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), osp.join(args.output, "best.pt"))
    print(f"✓ Training complete — best val acc = {best_acc:.2f}%")


if __name__ == '__main__':
    fix_seed(42)
    args = get_args_parser()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = float(args.lr)
    batch_size = args.batch
    freeze_backbone = args.freeze
    num_cls = 50
    optim_type = args.optim_type
    arch_ver = args.arch_ver
    output_dir = f'outputs/arch{arch_ver}_lr{lr}_freeze{"T" if freeze_backbone else "F"}_optim{optim_type}'
    if args.ver_name != "":
        output_dir += f"_V{args.ver_name}"
    args.output = output_dir
    ckpt_dir = osp.join(output_dir, 'ckpt')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(output_dir)

    optim_choices = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}
    model_choices = {'ver1': R34_ver1}

    train_annos, val_annos = split_trainval(num_train=45, num_val=10)

    img_size = 256
    crop_size = 224
    train_transform = T.Compose([
        T.Resize(img_size),
        T.RandAugment(),
        T.RandomResizedCrop(crop_size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMNET_MEAN, IMNET_STD),
        T.RandomErasing(p=0.1),
    ])
    val_transform = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(IMNET_MEAN, IMNET_STD),
    ])

    train_dataset = Proj_Dataset(train_annos, 'train', train_transform)
    val_dataset = Proj_Dataset(val_annos, 'val', val_transform)

    print("Train dataset: #", len(train_dataset))
    print("Val dataset: #", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    net = model_choices[arch_ver](num_classes=num_cls, freeze_backbone=args.freeze, dropout_p=0.3).to(device)

    model = net
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.classifier.parameters())
    opt = AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.01},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    steps_per_epoch = len(train_loader)
    sched = OneCycleLR(
        opt,
        max_lr=[args.lr * 0.01, args.lr],
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=25,
        final_div_factor=1e4,
    )
    crit = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    scaler = GradScaler()
    run_trainval(model, train_loader, val_loader, args, device, opt, sched, crit, scaler, writer)
