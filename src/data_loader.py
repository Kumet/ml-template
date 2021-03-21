from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


def get_data_loaders(data_dir, batch_size):
    dataset = MNIST(data_dir, download=True, transform=transforms.ToTensor())
    train, val = random_split(dataset, [55000, 5000])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
