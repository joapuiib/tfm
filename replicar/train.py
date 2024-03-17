import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
import torchvision

def resnet18(n_classes=2):
    model = torchvision.models.resnet18(pretrained='imagenet')
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=n_classes, bias=True)
    return model

def create_loaders(train_dataset, test_dataset, config):
    # Final loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                               batch_size=config.batch_size,
                                               num_workers=config.n_workers,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False,
                                              batch_size=config.batch_size,
                                              num_workers=config.n_workers,
                                              pin_memory=True)

    return train_loader, test_loader

def main(config):
    path = os.path.join(config.path, str(config.size))
    train_df = pd.read_csv(os.path.join(path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(path, 'test.csv'))

    # Create dataset
    train_df, test_df = create_dataset(config)

    # Loaders
    train_loader, test_loader = create_loaders(train_df, test_df, config)


    n_classes = len(train_df[config.target].unique())
    print(f'=> Training for {n_classes} classes')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data config
    parser.add_argument('--path', default=f'{os.path.expanduser("~")}/data/UNITOPATHO', type=str, help='UNITOPATHO dataset path')
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
