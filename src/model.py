import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=23, pretrained=True):
    """
    Returns a ResNet18 model modified for num_classes.
    """
    # Load the pretrained ResNet18
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Freeze the parameters if you only want to train the final layer
    # for param in model.parameters():
    #     param.requires_grad = False
        
    # Modify the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if __name__ == '__main__':
    # Test the model creation
    model = get_model(num_classes=23)
    print(model)
    
    # Test with dummy data
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
