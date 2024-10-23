import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


class VectorDataset(Dataset):
    def __init__(self, inputs, gpu=True):
        self.length = len(inputs)
        self.inputs = torch.tensor(inputs)
        if gpu and torch.cuda.is_available():
            self.inputs = self.inputs.cuda()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_vector = self.inputs[idx]
        return input_vector, [idx]


class Autoencoder(nn.Module):
    def __init__(self, emb_len=512):
        super().__init__()
        # encoder
        self.enc1 = nn.Linear(in_features=emb_len, out_features=256)
        self.enc2 = nn.Linear(in_features=256, out_features=128)
        self.enc3 = nn.Linear(in_features=128, out_features=50)
        # decoder
        self.dec1 = nn.Linear(in_features=50, out_features=128)
        self.dec2 = nn.Linear(in_features=128, out_features=256)
        self.dec3 = nn.Linear(in_features=256, out_features=emb_len)

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.sigmoid(self.enc3(x))
        return x

    def decode(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x

    def forward(self, x):
        # encoding
        x = self.encode(x)
        # decoding
        x = self.decode(x)
        return x


def train_autoencoder(autoencoder, train_loader, val_loader, gpu=True):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0001)
    val_losses = []
    device = 'cpu'
    if gpu:
        device = 'cuda'
    autoencoder.to(device)
    autoencoder.eval()
    with torch.no_grad():
        val_loss = 0
        val_count = 0
        for data, _ in val_loader:
            optimizer.zero_grad()
            output = autoencoder(data)
            loss = criterion(output, data)
            val_loss += loss.item() * len(data)
            val_count += len(data)

        val_loss /= val_count
        val_losses.append(val_loss)
        print(f'Epoch: {-1}, Val Loss: {val_loss}')

    for epoch in range(100):
        autoencoder.train()
        for data, _ in train_loader:
            optimizer.zero_grad()
            output = autoencoder(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
        autoencoder.eval()
        with torch.no_grad():
            val_loss = 0
            val_count = 0
            for data, _ in val_loader:
                optimizer.zero_grad()
                output = autoencoder(data)
                loss = criterion(output, data)
                val_loss += loss.item() * len(data)
                val_count += len(data)

            val_loss /= val_count
            val_losses.append(val_loss)
            print(f'Epoch: {epoch}, Val Loss: {val_loss}')

    print(output[:10], data[:10])

    return autoencoder


def make_loader(dataset_path, batch_size=64, shuffle=False):
    dataset = torch.load(dataset_path)
    dataset = VectorDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def save_checkpoint(autoencoder, save_path):
    autoencoder = autoencoder.to('cpu')
    state_dict = autoencoder.state_dict()
    torch.save(state_dict, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--val_path', type=str)
    parser.add_argument('--save_path', type=str, default='/mnt/bn/ai4s-hl/bamboo/hongyi/debug/checkpoints/autoencoder_04.pt')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    train_loader = make_loader(args.train_path, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(args.val_path, batch_size=args.batch_size, shuffle=False)
    autoencoder = Autoencoder()

    train_autoencoder(autoencoder, train_loader, val_loader, True)

    save_checkpoint(autoencoder, args.save_path)


if __name__ == '__main__':
    main()
