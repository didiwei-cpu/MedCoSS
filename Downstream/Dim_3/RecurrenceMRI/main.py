import argparse
import os
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataloader.Recurrence_MRI_dataset import RecurrenceMRIDataset
from model.Unimodel import Unified_Model


def get_parser():
    parser = argparse.ArgumentParser(description="Recurrence classification")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--label_file', type=str, default='CC_end.xlsx')
    parser.add_argument('--snapshot_dir', type=str, default='snapshots/tmp/')
    parser.add_argument('--reload_from_pretrained', action='store_true')
    parser.add_argument('--pretrained_path', type=str, default='checkpoint.pth')
    parser.add_argument('--input_size', type=str, default='8,64,64')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    return parser


def build_loader(args, split):
    crop = tuple(int(x) for x in args.input_size.split(','))
    ds = RecurrenceMRIDataset(args.data_root, split=split, label_file=args.label_file, crop_size=crop)
    shuffle = split == 'train'
    loader = DataLoader(ds, batch_size=args.batch_size if shuffle else 1,
                        shuffle=shuffle, num_workers=args.num_workers,
                        pin_memory=True)
    return loader, crop


def train_epoch(model, loader, optimizer):
    model.train()
    losses = []
    for vol, lab in loader:
        vol = vol.cuda(non_blocking=True)
        lab = lab.long().cuda(non_blocking=True)
        data = {"data": vol, "labels": lab, "modality": "3D image"}
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def evaluate(model, loader):
    model.eval()
    model.cal_acc = True
    prob_list, pred_list, gt_list = [], [], []
    with torch.no_grad():
        for vol, lab in loader:
            vol = vol.cuda(non_blocking=True)
            lab = lab.long().cuda(non_blocking=True)
            data = {"data": vol, "labels": lab, "modality": "3D image"}
            _, prob = model(data)
            prob = prob[:, 1].cpu().numpy()
            pred = (prob >= 0.5).astype(np.int64)
            prob_list.append(prob)
            pred_list.append(pred)
            gt_list.append(lab.cpu().numpy())
    model.cal_acc = False
    probs = np.concatenate(prob_list)
    preds = np.concatenate(pred_list)
    gts = np.concatenate(gt_list)
    auc = metrics.roc_auc_score(gts, probs)
    acc = metrics.accuracy_score(gts, preds)
    tp = np.sum((preds == 1) & (gts == 1))
    tn = np.sum((preds == 0) & (gts == 0))
    fp = np.sum((preds == 1) & (gts == 0))
    fn = np.sum((preds == 0) & (gts == 1))
    sen = tp / (tp + fn + 1e-8)
    spe = tn / (tn + fp + 1e-8)
    return acc, auc, sen, spe


def main():
    parser = get_parser()
    args = parser.parse_args()

    cudnn.benchmark = True
    train_loader, crop = build_loader(args, 'train')
    val_loader, _ = build_loader(args, 'val')

    model = Unified_Model(now_3D_input_size=crop, num_classes=2,
                          pre_trained=args.reload_from_pretrained,
                          pre_trained_weight=args.pretrained_path)
    model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=0.0001)

    best_auc = 0.0
    for epoch in range(args.num_epochs):
        loss = train_epoch(model, train_loader, optimizer)
        acc, auc, sen, spe = evaluate(model, val_loader)
        if auc > best_auc:
            best_auc = auc
            os.makedirs(args.snapshot_dir, exist_ok=True)
            torch.save({'model': model.state_dict(), 'epoch': epoch},
                       os.path.join(args.snapshot_dir, 'checkpoint.pth'))
        print(
            f"Epoch {epoch}: loss {loss:.4f}, acc {acc:.4f}, auc {auc:.4f}, sen {sen:.4f}, spe {spe:.4f}"
        )


if __name__ == '__main__':
    main()
