from resnet_pytorch import ResNet
import torch
import torchvision.transforms.v2 as transforms
import torchvision
import torch.utils.data.dataloader as dl
import matplotlib.pyplot as plt
import torch.jit as tj

TRAINING_DATA_DIR = './dataset/preprocessed/training'
TESTING_DATA_DIR = './dataset/preprocessed/testing'

runtime_preprocess = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.RandomRotation(1),
    transforms.RandomHorizontalFlip(.35),
    transforms.RandomGrayscale(.35),
])

def main():
    training_dataset = torchvision.datasets.ImageFolder(TRAINING_DATA_DIR, transform=runtime_preprocess)
    training_loader = dl.DataLoader(training_dataset, shuffle=True, num_workers=8, batch_size=100, pin_memory=True, persistent_workers=True)

    testing_dataset = torchvision.datasets.ImageFolder(TESTING_DATA_DIR, transform=runtime_preprocess)
    testing_loader = dl.DataLoader(testing_dataset, shuffle=True, num_workers=1, batch_size=10, pin_memory=True, persistent_workers=True)

    device_type = "cuda" if torch.cuda.is_available else "cpu"
    device = torch.device(device_type)
    print("Using " + device_type)


    model = ResNet.from_name('resnet18').to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

    cycle_error = list[float]()
    test_accuracy = list[float]()

    num_epochs = 50

    torch.manual_seed(0)

    for epoch in range(num_epochs):
        model.train(True)
        running_loss = 0

        for (i, (inputs, labels)) in enumerate(training_loader):
            optimizer.zero_grad(set_to_none=True)

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print(i, loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        cycle_error.append(running_loss)

        # Validating for given cycle
        with torch.no_grad():
            model.train(False)
            correct = 0
            total = 0
            for (inputs, labels) in testing_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                acc = correct / total
                test_accuracy.append(acc)
                print("Epoch: [{}/{}], Training loss: {:.4f}, Accuracy: {}".format(epoch+1,num_epochs,running_loss, acc))

    tj.save(tj.script(model), "./model.pt")

    plt.plot(range(1, len(cycle_error) + 1), cycle_error)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.plot(range(1, len(test_accuracy) + 1), test_accuracy)
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.show()

if __name__ == "__main__":
    main()