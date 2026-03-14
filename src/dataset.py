import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'New Plant Diseases Dataset')
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

# Data transformations
def get_transforms():
    # Define standard transforms
    data_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return data_transforms

def get_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found at {data_dir}")

    # Load dataset using ImageFolder
    transform = get_transforms()
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Calculate lengths for 70-15-15 split
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # Randomly split the dataset
    generator = torch.Generator().manual_seed(42) # for reproducibility
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
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
