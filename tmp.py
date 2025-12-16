from sklearn.cluster import KMeans 
import torchvision.transforms as transforms, torchvision, matplotlib.pyplot as plt
import torch 

trainset = torchvision.datasets.CIFAR10(root='./data', 
                                        train=True, 
                                        download=True,
                                        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=4, 
                                          shuffle=True)

images, labels = next(iter(trainloader))
# plt.imshow(torchvision.utils.make_grid(images).permute(1, 2, 0) / 2 + 0.5); 
# plt.title(' '.join(trainset.classes[label] for label in labels)); plt.show()

print(images.shape)
print(labels.shape)