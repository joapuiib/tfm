#!/bin/env python3

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
import torchvision

import albumentations
import albumentations.pytorch
from albumentations.pytorch import ToTensorV2
from functools import partial

import os
import argparse
import pandas as pd
import numpy as np
import random

from unitopatho import UnitopathoDataset
import utils

class UnitopathTrain():
    def __init__(self, config, checkpoint=None):
        self.config = config
        self.groupby = config.target
        self.checkpoint = None

        utils.set_seed(config.seed)
        self.scaler = torch.cuda.amp.GradScaler()


    def resnet18(self, n_classes=2):
        """
        ResNet18 model with custom final layer matching the number of classes
        :param n_classes: number of classes
        """
        model = torchvision.models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=n_classes, bias=True)
        return model


    def read_data(self):
        """
        Read data from csv files
        :param config: configuration object
        :return: train_df, test_df
        """
        self.path = os.path.join(self.config.path, str(self.config.size))
        train_df = pd.read_csv(os.path.join(self.path, 'train.csv'))
        test_df = pd.read_csv(os.path.join(self.path, 'test.csv'))

        print('=> Target attribute:', self.config.target)
        print('=> Raw data (train)')
        print(train_df.groupby(self.groupby).count())

        print('\n=> Raw data (test)')
        print(test_df.groupby(self.groupby).count())

        self.train_df = train_df
        self.test_df = test_df


    def preprocess_data(self):
        # TODO: Entendre aquesta funció
        """
        Sets the train and test dataframes after preprocessing

        :param train_df: train dataframe
        :param test_df: test dataframe
        :param config: configuration object
        """
        train_df = self.train_df
        test_df = self.test_df
        config = self.config

        if config.target == 'grade':
            # Preprocess data
            train_df = self.preprocess_df(train_df, config.label)
            test_df = self.preprocess_df(test_df, config.label)

            print('\n=> Preprocessed data (train)')
            print(train_df.groupby(self.groupby).count())

            print('\n=> Preprocessed data (test)')
            print(test_df.groupby(self.groupby).count())

            # balance train_df (sample mean size)
            groups = train_df.groupby('grade').count()
            grade_min = int(groups.image_id.idxmin())
            mean_size = int(train_df.groupby('grade').count().mean()['image_id'])

            train_df = pd.concat((
                train_df[train_df.grade == 0].sample(mean_size, replace=(grade_min==0), random_state=config.seed).copy(),
                train_df[train_df.grade == 1].sample(mean_size, replace=(grade_min==1), random_state=config.seed).copy()
            ))
        else:
            # balance train_df (sample 3rd min_size)
            # TODO: No entenc aquest cas
            min_size = np.sort(train_df.groupby(self.config.target).count()['image_id'])[2]
            train_df = train_df.groupby(self.config.target).apply(lambda group: group.sample(min_size, replace=len(group) < min_size, random_state=config.seed)).reset_index(drop=True)

        self.train_df = train_df
        self.test_df = test_df
        self.n_classes = len(self.train_df[self.config.target].unique())


    def preprocess_df(self, df, label):
        # TODO: No entenc aquest cas
        if label == 'norm':
            # Passa a -1 tots els graus LG
            df.loc[df.grade == 0, 'grade'] = -1
            # Passa a 0 tots els tipus NORM
            df.loc[df.type == 'norm', 'grade'] = 0

        # Elimina totes les files amb grau -1
        df = df[df.grade >= 0].copy()

        # Filtra per tipus en cas de TA o TVA
        if label == 'ta' or label == 'tva':
            df = df[df.type == label].copy()
        return df


    def print_data_summary(self):
        train_df = self.train_df
        test_df = self.test_df

        print('\n---- DATA SUMMARY (After balance) ----')
        print('---------------------------------- Train ----------------------------------')
        print(train_df.groupby(self.groupby).count())
        print(len(train_df.wsi.unique()), 'WSIs')

        print('\n---------------------------------- Test ----------------------------------')
        print(test_df.groupby(self.groupby).count())
        print(len(test_df.wsi.unique()), 'WSIs')

        print(f'=> Training for {self.n_classes} classes')


    def create_model_params(self):
        im_mean, im_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # ImageNet
        norm = dict(
            rgb=dict(mean=im_mean,
                        std=im_std),
            he=dict(mean=im_mean,
                    std=im_std),
            gray=dict(mean=[0.5],
                        std=[1.0])
        )

        self.mean, self.std = norm[self.config.preprocess]['mean'], norm[self.config.preprocess]['std']
        print('=> mean, std:', self.mean, self.std)


    def set_data_augmentation(self):
        self.T_aug = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(90, p=0.5)
        ])

        self.T_jitter = albumentations.ColorJitter()

        self.T_tensor = ToTensorV2()

        self.T_post = albumentations.Compose([
            albumentations.Normalize(self.mean, self.std),
            self.T_tensor
        ])


    def apply_transforms(self, train, img):
        def normalize_he(x):
            if config.preprocess == 'he':
                img = x
                try:
                    img = T_tensor(image=img)['image']*255
                    img, _, _ = normalizer.normalize(img, stains=False)
                    img = img.numpy().astype(np.uint8)
                except Exception as e:
                    print('Could not normalize image:', e)
                    img = x
                return img
            return x

        img = normalize_he(img)
        if train:
            img = self.T_aug(image=img)['image']
            if self.config.preprocess == 'rgb':
                img = self.T_jitter(image=img)['image']
        x = img
        return self.T_post(image=x)['image']


    def apply_data_augmentation(self):
        self.T_train = partial(self.apply_transforms, True)
        self.T_test = partial(self.apply_transforms, False)


    def create_loaders(self):

        datasets_kwargs = {
            'path': self.path,
            'subsample': self.config.subsample,
            'target': self.config.target,
            'gray': self.config.preprocess == 'gray',
            'mock': self.config.mock
        }

        train_dataset = UnitopathoDataset(self.train_df, T=self.T_train, **datasets_kwargs)
        test_dataset = UnitopathoDataset(self.test_df, T=self.T_test, **datasets_kwargs)

        # Final loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                                   batch_size=config.batch_size,
                                                   num_workers=config.n_workers,
                                                   pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False,
                                                  batch_size=config.batch_size,
                                                  num_workers=config.n_workers,
                                                  pin_memory=True)

        self.train_loader = train_loader
        self.test_loader = test_loader


    def train(self):
        config = self.config
        train_loader = self.train_loader
        test_loader = self.test_loader

        n_channels = {
            'rgb': 3,
            'he': 3,
            'gray': 1
        }

        model = self.resnet18(n_classes=self.n_classes)
        model.conv1 = torch.nn.Conv2d(n_channels[config.preprocess], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if self.checkpoint is not None:
            model.load_state_dict(self.checkpoint['model'])
        model = model.to(config.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        criterion = F.cross_entropy

        for epoch in range(config.epochs):
            if config.test is None:
                train_metrics = utils.train(model, train_loader, criterion,
                                            optimizer, config.device, metrics=utils.metrics,
                                            accumulation_steps=config.accumulation_steps, scaler=self.scaler)
                scheduler.step()

            test_metrics = utils.test(model, test_loader, criterion, config.device, metrics=utils.metrics)

            if config.test is None:
                print(f'Epoch {epoch}: train: {train_metrics}')
                # wandb.log({'train': train_metrics, 'test': test_metrics})
                # torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config}, os.path.join(wandb.run.dir, 'model.pt'))

            print(f'test: {test_metrics}')
            if config.test is not None:
                break



def main(config):
    trainer = UnitopathTrain(config)
    # Read data
    trainer.read_data()

    # Preprocess data
    trainer.preprocess_data()

    trainer.print_data_summary()


    # Create loaders
    trainer.create_model_params()
    trainer.set_data_augmentation()
    trainer.apply_data_augmentation()

    trainer.create_loaders()
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data config
    parser.add_argument('--path', default=f'{os.path.expanduser("~")}/unitopath/', type=str, help='UNITOPATHO dataset path')
    parser.add_argument('--size', default=100, type=int, help='patch size in µm (default 100)')
    parser.add_argument('--subsample', default=-1, type=int, help='subsample size for data (-1 to disable, default -1)')

    # optimizer & network config
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--accumulation_steps', default=1, type=int, help='gradient accumulation steps')
    parser.add_argument('--n_workers', default=8, type=int)
    parser.add_argument('--architecture', default='resnet18', help='resnet18, resnet50, densenet121')

    # training config
    parser.add_argument('--preprocess', default='rgb', help='preprocessing type, rgb, he or gray. Default: rgb')
    # grade: distinguish between High Grade (1) and Low Grade (0)
    # type: distinguish between NORM, HP, TA and TVA
    # top_label: distinguish between NORM, HP, TA.HG, TA.LG, TVA.HG, TVA.LG
    parser.add_argument('--target', default='grade', help='target attribute: grade, type, top_label (default: grade)')
    parser.add_argument('--label', default='both', type=str, help='only when target=grade; values: ta, tva, norm or both (default: both)')
    parser.add_argument('--test', type=str, help='Run id to test', default=None)

    # misc config
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--mock', action='store_true', dest='mock', help='mock dataset (random noise)')
    parser.add_argument('--seed', type=int, default=42)
    parser.set_defaults(mock=False)

    config = parser.parse_args()
    config.device = torch.device(config.device)

    main(config)
