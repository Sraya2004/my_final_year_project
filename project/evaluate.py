import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import ACN

def evaluate():
    batch_size = 32
    num_classes = 10

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.FakeData(transform=transform)  # Replace with real dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ACN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('models/best_model.pth'))  # Load your saved model
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    evaluate()
