import argparse
import torch
import numpy as np
import torch.nn as nn
import os

from tqdm import tqdm
from dataloader import KFoldDataModule
from model import ChemBERT
from util import LossCalculator, huber_loss, mse_threshold_no_learn, ThresholdPenaltyLoss, MSLELoss

os.environ["WANDB_DISABLED"] = "true"
tqdm.pandas()
torch.manual_seed(12345)


def parse_args():
    # argparse로 커맨드 라인 인자 받기
    parser = argparse.ArgumentParser(description="KFold DataModule for PyTorch Lightning")

    # 각 인자를 argparse로 정의
    parser.add_argument('--train_df', type=str, default='../dataset/train.csv',
                        help='Path to the train dataset CSV')
    parser.add_argument('--k_idx', type=int, default=1, help='Fold index')
    parser.add_argument('--num_split', type=int, default=5, help='Number of folds')
    parser.add_argument('--split_seed', type=int, default=41, help='Random seed for splitting data')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--persistent_workers', type=bool, default=False, help='persistent_workers')
    parser.add_argument('--pin_memory', action='store_true', help='Use pin memory for data loading')
    parser.add_argument('--criterion', default='mse', help='mse(default), msle, huber, threshold_penalty, threadshold_nolearn')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--steplr', type=int, default=5, help='step lr')
    parser.add_argument('--epoch', type=int, default=60, help='epoch')

    # 파싱된 인자를 args에 저장
    args = parser.parse_args()

    return args


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    args = parse_args()

    epochs = args.epoch

    datamodule = KFoldDataModule(args)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    model = ChemBERT(
        out_dim=1,
        fp_dim=0,
        max_chemberta_len=1,
        max_graphormer_len=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print('learning rate:', args.learning_rate)

    if args.criterion == 'threshold_penalty':
        loss = ThresholdPenaltyLoss(threshold=0.5, penalty_weight=0.1)
    elif args.criterion == 'threshold_nolearn':
        loss = mse_threshold_no_learn
    elif args.criterion == 'huber':  # 오차 없이 예측해야 하므로 별로임
        loss = huber_loss
    elif args.criterion == 'msle':
        loss = MSLELoss()
    else:
        loss = nn.MSELoss()

    loss_calculator = LossCalculator(
        criterion=loss
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.steplr, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    model.to(DEVICE)

    for epoch in range(1, epochs + 1):
        # Training loop
        model.train()
        loss_calculator.epoch(epoch)

        for batch in tqdm(train_loader, ncols=75):
            optimizer.zero_grad(set_to_none=True)
            batch = batch.to(DEVICE)
            prediction = model(batch)

            prediction = prediction.type(torch.float)
            y = batch.y.type(torch.float)

            loss = loss_calculator(prediction, y)
            loss.backward()
            optimizer.step()

        scheduler.step()
        loss_calculator.print_status()

        # Validation loop
        model.eval()
        loss_calculator.epoch(epoch)

        with torch.no_grad():
            for batch in tqdm(val_loader, ncols=75):
                batch = batch.to(DEVICE)
                prediction = model(batch)

                prediction = prediction.type(torch.float)
                y = batch.y.type(torch.float)

                loss_calculator(prediction, y)

        loss_calculator.print_status(validation=True)


if __name__ == '__main__':
    main()
