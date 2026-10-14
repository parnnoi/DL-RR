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

DATA_DIR = '/storage/slurm/home/68070701616@cpe.kmutt.ac.th/DL-RR/src/data'
RESULTS_FILE = f'../results/{DATASET}_e{EPOCHS}_b{BATCH_SIZE}_s{IMAGE_SIZE}.csv'
LOSS_FILE = RESULTS_FILE.replace('.csv', '_loss.csv')
LOSS_PLOT = RESULTS_FILE.replace('.csv', '_loss.png')

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models.vision_transformer import interpolate_embeddings
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Notebook-defined datasets cannot be sent to worker processes on macOS
NUM_WORKERS = 0 if sys.platform == 'darwin' else 4

print('torch', torch.__version__, '| device:', device)

def load_moeimouto():
    train, test = [], []
    folders = sorted(f for f in Path(DATA_DIR, 'moeimouto-faces').iterdir() if f.is_dir())
    for label, folder in enumerate(folders):
        files = sorted(f for f in folder.glob('*.png') if not f.name.startswith('._'))
        files = [f for f in files if Image.open(f).mode == 'RGB']
        n_train = int(0.8 * len(files))
        train += [(f, label) for f in files[:n_train]]
        test += [(f, label) for f in files[n_train:]]
    columns = ['path', 'label']
    return pd.DataFrame(train, columns=columns), None, pd.DataFrame(test, columns=columns)


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

print('classes:', NUM_CLASSES)
print('train:', len(train_df), '| val:', 0 if val_df is None else len(val_df), '| test:', len(test_df))

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize,
])


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


def make_loader(df, transform, shuffle):
    return DataLoader(FaceDataset(df, transform), batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS)


train_loader = make_loader(train_df, train_transform, shuffle=True)
val_loader = None if val_df is None else make_loader(val_df, test_transform, shuffle=False)
test_loader = make_loader(test_df, test_transform, shuffle=False)

class ShallowNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        size = ((IMAGE_SIZE - 4) // 2 - 4) // 2    # feature map size after the two conv + pool blocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * size * size, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


RESNETS = {
    'R-18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    'R-152': (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1),
}

# name: (builder, pretrained weights, patch size)
VITS = {
    'B-16': (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1, 16),
    'B-32': (models.vit_b_32, models.ViT_B_32_Weights.IMAGENET1K_V1, 32),
    'L-16': (models.vit_l_16, models.ViT_L_16_Weights.IMAGENET1K_V1, 16),
    'L-32': (models.vit_l_32, models.ViT_L_32_Weights.IMAGENET1K_V1, 32),
}


def build_resnet(name, pretrained):
    builder, weights = RESNETS[name]
    model = builder(weights=weights if pretrained else None)
    if pretrained:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def build_vit(name, pretrained):
    builder, weights, patch_size = VITS[name]
    model = builder(image_size=IMAGE_SIZE)
    if pretrained:
        # The weights were trained on 224x224 images: resize the position embeddings to IMAGE_SIZE
        state = interpolate_embeddings(IMAGE_SIZE, patch_size, weights.get_state_dict(progress=True))
        model.load_state_dict(state)
    model.heads = nn.Linear(model.hidden_dim, NUM_CLASSES)
    return model


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


def train(model):
    """Trains the model and returns its training loss at each epoch."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_EVERY, gamma=LR_DECAY)
    criterion = nn.CrossEntropyLoss()

    epoch_losses = []
    for epoch in range(1, EPOCHS + 1):
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
        print(message)

    return epoch_losses

results = []
losses = {}    # run name -> training loss at each epoch

for pretrained in PRETRAINED:
    for name in MODELS:
        if name == 'SN' and pretrained:
            continue
        run = f'{name} (pretrained)' if pretrained else name
        print(f'--- {run} ---')
        model = build_model(name, pretrained).to(device)
        losses[run] = train(model)
        top1, top5 = evaluate(model, test_loader)
        print(f'test top-1 {top1:.2f} | test top-5 {top5:.2f}')

        results.append({'Model': name, 'Pretrained': pretrained, 'Top-1': top1, 'Top-5': top5})
        pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
        pd.DataFrame(losses, index=pd.RangeIndex(1, EPOCHS + 1, name='Epoch')).to_csv(LOSS_FILE)

table = pd.DataFrame(results).pivot(index='Model', columns='Pretrained', values=['Top-1', 'Top-5'])
table = table.swaplevel(axis=1).sort_index(axis=1).reindex(MODELS)
table.round(2)

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
plt.show()
