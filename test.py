import torch
from net import MyLeNet5
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

data_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size =16, shuffle = True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size =16, shuffle = True)

device = "cuda" if torch.cuda.is_available() else 'cpu'

model = MyLeNet5().to(device)

model.load_state_dict(torch.load("D:/CNN_MNIST/save_model/best_model.pth"))

classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

show = ToPILImage()

for i in range(20):
    X,  y =test_dataset[i][0], test_dataset[i][1]
    show(X).show()

    X = Variable(torch.unsqueeze(X,dim = 0).float(),requires_grad=False).to(device)
    with torch.no_grad():
        pred = model(X)

        predicted, actual = classes[torch.argmax(pred[0])], classes[y]

        print(f'predicted: "{predicted}", actual: "{actual}"')