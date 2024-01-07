from resnet_pytorch import ResNet
import torch
import torchvision.transforms as transforms
import torchvision
import torch.utils.data.dataloader as dl

TRAINING_DATA_DIR = './dataset/training'
TESTING_DATA_DIR = './dataset/testing'

model = ResNet.from_name('resnet18')

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize()
])

training_dataset = torchvision.datasets.ImageFolder(TRAINING_DATA_DIR, transform=preprocess)
training_loader = dl.DataLoader(training_dataset, shuffle=True, num_workers=0, batch_size=20, pin_memory=True)

testing_dataset = torchvision.datasets.ImageFolder(TESTING_DATA_DIR, transform=preprocess)
testing_loader = dl.DataLoader(testing_dataset, shuffle=True, num_workers=0, batch_size=20, pin_memory=True)

cycle_error = list[float]()
test_accuracy = list[float]()

num_epochs = 20

for epoch in range(num_epochs):
    model.train(True)
    running_loss = 0