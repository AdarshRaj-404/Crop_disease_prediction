import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset

# Constants
DATA_DIR = r"c:\Users\WIN 10\Downloads\CROP DIS DATASET COMPRESSED (8)"
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

# Data transformations
def get_train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_eval_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class SubsetDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def get_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found at {data_dir}")

    # Load dataset using ImageFolder without transforms to allow SubsetDataset to apply them
    full_dataset = datasets.ImageFolder(root=data_dir, transform=None)
    
    # Calculate lengths for 70-15-15 split
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # Randomly split the dataset
    generator = torch.Generator().manual_seed(42) # for reproducibility
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Apply specific transforms
    train_dataset = SubsetDataset(train_subset, transform=get_train_transforms())
    val_dataset = SubsetDataset(val_subset, transform=get_eval_transforms())
    test_dataset = SubsetDataset(test_subset, transform=get_eval_transforms())
    
    # Create DataLoaders
    # Reduced num_workers to 0 to prevent Windows multi-processing overhead and DataLoader freeze issues
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    class_names = full_dataset.classes
    
    return train_loader, val_loader, test_loader, class_names

if __name__ == '__main__':
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get a batch of training data
    inputs, classes_batch = next(iter(train_loader))
    print(f"Input shape: {inputs.shape}")
    print(f"Classes shape: {classes_batch.shape}")
