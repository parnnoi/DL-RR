DATASET = 'moeImouto'    # 'moeImouto' or 'DAFre'
EPOCHS = 200             # 50 or 200
BATCH_SIZE = 64          # 64 or 1024
IMAGE_SIZE = 128         # 128 or 224

MODELS = ['R-18', 'R-152', 'B-16', 'B-32', 'L-16', 'L-32']    # 'SN' is also available
PRETRAINED = [False, True]                                    # train from scratch, then from ImageNet weights

# Same for every experiment in the paper
LEARNING_RATE = 0.001
MOMENTUM = 0.9
LR_DECAY_EVERY = 20 if EPOCHS == 50 else 50    # epochs between two LR decays
LR_DECAY = 1 / 3                               # the LR is divided by 3 at each decay
SEED = 0                                       # same default as the authors' code (Animesion)

DATA_DIR = '/storage/slurm/home/68070701616@cpe.kmutt.ac.th/DL-RR/src/data'
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Everything of one experiment goes to results/<experiment>/, next to src/ (wherever the script is started from).
# '_animesion' = models, data split and pre-processing of the authors' code at the time of the paper (Animesion, Jan 2021)
EXPERIMENT = f'{DATASET}_e{EPOCHS}_b{BATCH_SIZE}_s{IMAGE_SIZE}_animesion'
SPLITS_DIR = Path(__file__).resolve().parent / 'splits'    # the authors' moeImouto train / test split
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results' / EXPERIMENT
CHECKPOINT_DIR = RESULTS_DIR / 'checkpoints'    # progress of the model being trained, saved every epoch
MODELS_DIR = RESULTS_DIR / 'models'             # finished models
RESULTS_FILE = RESULTS_DIR / 'results.csv'
LOSS_FILE = RESULTS_DIR / 'loss.csv'
LOSS_PLOT = RESULTS_DIR / 'loss.png'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from pytorch_pretrained_vit import ViT
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# macOS starts data workers by re-running this whole script, so load the data in the main process there
NUM_WORKERS = 0 if sys.platform == 'darwin' else 4


def log(message=''):
    """Prints the message with the current time, straight to the Slurm log."""
    print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}', flush=True)


def duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h {minutes:02d}m {seconds:02d}s'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_rng_state():
    """The state of every random generator, saved in the checkpoints so a resumed run continues the same random sequence."""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all(),
    }


def set_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


START_TIME = time.time()
set_seed(SEED)

log('=' * 70)
log(f'START experiment {EXPERIMENT}')
log(f'dataset {DATASET} | epochs {EPOCHS} | batch size {BATCH_SIZE} | image size {IMAGE_SIZE} | seed {SEED}')
gpu = f' ({torch.cuda.get_device_name(0)})' if device == 'cuda' else ' (no GPU found: training will be very slow)'
log(f'torch {torch.__version__} | device {device}{gpu}')
log(f'results folder: {RESULTS_DIR}')
log('loading the data ...')

def load_moeimouto():
    """The authors' split files: 0.8 / 0.2 random split, the 2 non-RGB images already removed."""
    root = Path(DATA_DIR, 'moeimouto-faces')

    def read(name):
        df = pd.read_csv(SPLITS_DIR / 'moeImouto' / name, names=['label', 'path'])
        df['path'] = [root / p for p in df['path']]
        return df

    return read('train.csv'), None, read('test.csv')


def load_dafre():
    root = Path(DATA_DIR, 'daf')

    def read(name):
        df = pd.read_csv(root / 'labels' / name, names=['label', 'path'])
        df['path'] = [root / 'faces' / p for p in df['path']]
        return df

    train_val, val, test = read('train.csv'), read('val.csv'), read('test.csv')
    train = train_val[~train_val['path'].isin(val['path'])]
    return train, val, test


train_df, val_df, test_df = load_moeimouto() if DATASET == 'moeImouto' else load_dafre()
NUM_CLASSES = train_df['label'].nunique()

log(f'data loaded: {NUM_CLASSES} classes | train {len(train_df)} | val {0 if val_df is None else len(val_df)} | test {len(test_df)}')

def make_transforms(name):
    """Pre-processing of the authors' code. Not in the paper: the color jitter, and the ImageNet normalization for ResNets."""
    if name in RESNETS:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, test_transform


class FaceDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df['path'].tolist()
        self.labels = df['label'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert('RGB')
        return self.transform(image), self.labels[i]


def make_loaders(name):
    """Returns the train, val (None for moeImouto) and test loaders, with the pre-processing of this model."""
    train_transform, test_transform = make_transforms(name)

    def loader(df, transform, shuffle):
        return DataLoader(FaceDataset(df, transform), batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS)

    val_loader = None if val_df is None else loader(val_df, test_transform, shuffle=False)
    return loader(train_df, train_transform, True), val_loader, loader(test_df, test_transform, False)

class ShallowNet(nn.Module):
    """The authors' 5-layer CNN: 2 conv blocks and 2 fully-connected layers."""

    def __init__(self, num_classes):
        super().__init__()
        size = ((IMAGE_SIZE - 4) // 2 - 4) // 2    # feature map size after the two conv + pool blocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * size * size, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


RESNETS = {
    'R-18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    'R-152': (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1),
}

# The authors' ViT (dropout 0.1). Pretrained = Google's ImageNet-21k weights, downloaded and converted on first use.
VITS = {'B-16': 'B_16', 'B-32': 'B_32', 'L-16': 'L_16', 'L-32': 'L_32'}


def build_resnet(name, pretrained):
    builder, weights = RESNETS[name]
    model = builder(weights=weights if pretrained else None)
    if pretrained:
        for parameter in model.parameters():
            parameter.requires_grad = False
    else:
        # Same initialization as the authors' code
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.normal_(module.bias, std=1e-6)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def build_vit(name, pretrained):
    # The library resizes the position embeddings to IMAGE_SIZE and drops the 21k-class head by itself
    return ViT(VITS[name], pretrained=pretrained, image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)


def build_model(name, pretrained):
    if name == 'SN':
        return ShallowNet(NUM_CLASSES)
    if name in RESNETS:
        return build_resnet(name, pretrained)
    return build_vit(name, pretrained)

@torch.no_grad()
def evaluate(model, loader):
    """Returns the top-1 and top-5 accuracy (%)."""
    model.eval()
    top1, top5 = 0, 0
    for images, labels in loader:
        best5 = model(images.to(device)).cpu().topk(5, dim=1).indices
        top1 += (best5[:, 0] == labels).sum().item()
        top5 += (best5 == labels[:, None]).any(dim=1).sum().item()
    total = len(loader.dataset)
    return 100 * top1 / total, 100 * top5 / total


def save(data, path):
    """Saves to a temporary file first, so a job killed while saving never leaves a broken file."""
    temporary = path.with_suffix('.tmp')
    torch.save(data, temporary)
    os.replace(temporary, path)


def train(model, checkpoint_file, train_loader, val_loader):
    """Trains the model and returns its training loss at each epoch.

    The progress is saved to checkpoint_file after every epoch. If that file already exists,
    training continues from the last saved epoch instead of starting again.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_EVERY, gamma=LR_DECAY)
    criterion = nn.CrossEntropyLoss()

    start_epoch, epoch_losses = 1, []
    if checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if 'rng' in checkpoint:    # checkpoints made before the seed was added do not have it
            set_rng_state(checkpoint['rng'])
        start_epoch = checkpoint['epoch'] + 1
        epoch_losses = checkpoint['losses']
        log(f'RESUME from the checkpoint: continuing at epoch {start_epoch}/{EPOCHS}')
    else:
        log(f'START training from epoch 1/{EPOCHS}')

    train_start = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            loss = criterion(model(images.to(device)), labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        epoch_losses.append(total_loss / len(train_loader))

        message = f'epoch {epoch}/{EPOCHS} | loss {epoch_losses[-1]:.4f}'
        if val_loader is not None:
            val_top1, val_top5 = evaluate(model, val_loader)
            message += f' | val top-1 {val_top1:.2f} | val top-5 {val_top5:.2f}'

        save({
            'epoch': epoch,
            'losses': epoch_losses,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'rng': get_rng_state(),
        }, checkpoint_file)

        time_per_epoch = (time.time() - train_start) / (epoch - start_epoch + 1)
        message += f' | {duration(time.time() - epoch_start)} | model left {duration(time_per_epoch * (EPOCHS - epoch))}'
        log(message + ' | checkpoint saved')

    log(f'training finished in {duration(time.time() - train_start)} (this job)')
    return epoch_losses

runs = [(name, pretrained) for pretrained in PRETRAINED for name in MODELS if not (name == 'SN' and pretrained)]


def file_name(name, pretrained):
    return f'{name}_pretrained.pt' if pretrained else f'{name}_scratch.pt'


# Overview of what is already done, so a re-run shows where it picks up
done = [run for run in runs if (MODELS_DIR / file_name(*run)).exists()]
in_progress = [run for run in runs if (CHECKPOINT_DIR / file_name(*run)).exists()]
log(f'{len(runs)} models to train: {len(done)} finished, {len(in_progress)} in progress, '
    f'{len(runs) - len(done) - len(in_progress)} not started')

results = []
losses = {}    # run name -> training loss at each epoch

for i, (name, pretrained) in enumerate(runs, 1):
    run = f'{name} (pretrained)' if pretrained else name
    checkpoint_file = CHECKPOINT_DIR / file_name(name, pretrained)
    model_file = MODELS_DIR / file_name(name, pretrained)
    log('-' * 70)
    log(f'[{i}/{len(runs)}] {run}')

    if model_file.exists():
        finished = torch.load(model_file, map_location='cpu')
        log(f'already finished, skipping (test top-1 {finished["top1"]:.2f} | test top-5 {finished["top5"]:.2f})')
    else:
        set_seed(SEED)    # every model starts from the same seed, whatever ran before it
        log('building the model' + (' with its ImageNet weights' if pretrained else ''))
        model = build_model(name, pretrained).to(device)
        train_loader, val_loader, test_loader = make_loaders(name)
        epoch_losses = train(model, checkpoint_file, train_loader, val_loader)

        log('testing ...')
        top1, top5 = evaluate(model, test_loader)
        finished = {'model': model.state_dict(), 'losses': epoch_losses, 'top1': top1, 'top5': top5}
        save(finished, model_file)
        checkpoint_file.unlink()    # the model is finished: its progress file is not needed anymore
        log(f'FINISHED {run}: test top-1 {top1:.2f} | test top-5 {top5:.2f}')
        log(f'model saved to {model_file} | checkpoint deleted')

    losses[run] = finished['losses']
    results.append({'Model': name, 'Pretrained': pretrained, 'Top-1': finished['top1'], 'Top-5': finished['top5']})
    pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
    pd.DataFrame(losses, index=pd.RangeIndex(1, EPOCHS + 1, name='Epoch')).to_csv(LOSS_FILE)

log('-' * 70)
log('ALL MODELS FINISHED. Test accuracy (%):')
table = pd.DataFrame(results).pivot(index='Model', columns='Pretrained', values=['Top-1', 'Top-5'])
table = table.swaplevel(axis=1).sort_index(axis=1).reindex(MODELS)
print(table.round(2), flush=True)

loss_table = pd.DataFrame(losses, index=pd.RangeIndex(1, EPOCHS + 1, name='Epoch'))
compare = len(losses) > 1

fig, axes = plt.subplots(1, 2 if compare else 1, figsize=(14 if compare else 8, 5), squeeze=False)
curves_ax = axes[0, 0]

colors = {name: f'C{i}' for i, name in enumerate(MODELS)}
for run in loss_table.columns:
    name = run.split(' ')[0]
    style = '--' if 'pretrained' in run else '-'
    curves_ax.plot(loss_table.index, loss_table[run], style, color=colors[name], label=run)
curves_ax.set_title(f'Training loss on {DATASET}')
curves_ax.set_xlabel('Epoch')
curves_ax.xaxis.get_major_locator().set_params(integer=True)
curves_ax.set_ylabel('Cross-entropy loss')
curves_ax.grid(alpha=0.3)
curves_ax.legend()

if compare:
    final = loss_table.iloc[-1].sort_values()
    bars_ax = axes[0, 1]
    bars_ax.barh(final.index, final.values, color=[colors[run.split(' ')[0]] for run in final.index])
    bars_ax.invert_yaxis()    # lowest loss on top
    bars_ax.set_title(f'Training loss at the last epoch ({EPOCHS})')
    bars_ax.set_xlabel('Cross-entropy loss')
    bars_ax.set_xlim(0, final.max() * 1.15)    # room for the value labels
    for y, value in enumerate(final.values):
        bars_ax.text(value, y, f' {value:.3f}', va='center')

fig.tight_layout()
fig.savefig(LOSS_PLOT, dpi=150)
log(f'saved {RESULTS_FILE.name}, {LOSS_FILE.name} and {LOSS_PLOT.name} to {RESULTS_DIR}')
log(f'DONE in {duration(time.time() - START_TIME)} (this job)')
