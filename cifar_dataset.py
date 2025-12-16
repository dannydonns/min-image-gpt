# file containing the cifar dataset class and tokenizer
import torch
import torch.nn
from torch.utils.data import Dataset
from sklearn.cluster import KMeans 
import torchvision.transforms as transforms, torchvision, matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader
import numpy as np
import joblib
import math

# plt.imshow(torchvision.utils.make_grid(images).permute(1, 2, 0) / 2 + 0.5); 
# plt.title(' '.join(trainset.classes[label] for label in labels)); plt.show()

# print(images.shape)
# print(labels.shape)

# images1 = images.permute(0, 2, 3, 1)
# print(images1.shape)
# images1 = images1.view(images1.shape[0], -1, images1.shape[3])
# print(images1.shape)

# pixels = images1.contiguous().view(-1, images1.shape[2])
# pixels = np.array(pixels)

# print(pixels.shape)



# print("Running K-means")
# kmeans = KMeans(n_clusters=512, random_state=0, n_init="auto").fit(pixels[:5120000])
# print("Done")

# filename = 'kmeans.joblib'
# # joblib.dump(kmeans, filename)
# kmeans = joblib.load(filename)
# print(pixels[0:100])
# print(kmeans.predict(pixels[50:51]))

# # Get the cluster assignments for each data point
# cluster_assignments = kmeans.labels_
# print(f"Cluster assignments for each data point: {cluster_assignments}")
# for i in np.sort(np.unique(cluster_assignments)):
#     print(i)

# # Get the coordinates of the cluster centroids
# cluster_centroids = kmeans.cluster_centers_
# print(f"Coordinates of cluster centroids:\n{cluster_centroids}")

# # Example: Accessing the centroid for a specific cluster number
# # If you want the coordinates of cluster 0:
# centroid_of_cluster_0 = cluster_centroids[511]
# print(f"Centroid of cluster 0: {centroid_of_cluster_0}")

class PixelTokenizer:
    def __init__(self, filename='kmeans.joblib'):
        # 1. Load the pre-trained Scikit-Learn KMeans model
        # Ensure 'kmeans.joblib' exists or pass the correct path
        self.kmeans = joblib.load(filename)
        
        # 2. Extract Centroids
        # We convert them to a PyTorch buffer immediately for easier lookup later
        # shape: (512, 3)
        self.cluster_centroids = torch.from_numpy(self.kmeans.cluster_centers_).float()

    def tokenize(self, images):
        """
        Input:  Tensor (N, S, 3) - Flattened sequence of pixels
        Output: Tensor (N, S)    - Sequence of cluster IDs (0-511)
        """
        im1, im2, im3, im4 = images.shape
        images = images.permute(0, 2, 3, 1)
        images = images.view(im1, im3*im4, -1)

        # Scikit-learn operates on NumPy arrays on the CPU
        # We must flatten the batch and sequence dimensions to (N*S, 3) for prediction
        flat_pixels = images.contiguous().view(-1, 3).cpu().numpy()
        
        # Predict the nearest cluster for each pixel
        cluster_ids = self.kmeans.predict(flat_pixels)
        
        # Convert back to PyTorch tensor and restore shape (N, S)
        tokens = torch.from_numpy(cluster_ids).view(images.shape[0], images.shape[1])
        
        return tokens.long() # Return as integer tokens

    def detokenize(self, tokens):
        """
        Input:  Tensor (N, S)    - Sequence of tokens
        Output: Tensor (N, 3, 32, 32) - Reconstructed images (CHW format)
        """
        # 1. Ensure centroids are on the same device as the input tokens
        centroids = self.cluster_centroids.to(tokens.device)
        
        # 2. Embedding Lookup / De-quantization
        # We use the token indices to look up the RGB values in the centroid table.
        # Input: (N, S) -> Output: (N, S, 3)
        reconstructed_pixels = centroids[tokens]
        
        # 3. Reshape to Image Dimensions
        # The user requested (N, 3, 32, 32). 
        # We assume S = 1024 (32*32). 
        N, S, C = reconstructed_pixels.shape
        H = W = int(math.sqrt(S)) # Should be 32
        
        # Current: (N, S, 3) -> Reshape to spatial: (N, 32, 32, 3)
        # Note: We use 32, 32 because the sequence S usually comes from flattening row-by-row
        images = reconstructed_pixels.view(N, H, W, C)
        
        # 4. Permute to PyTorch Channel-First format (N, C, H, W)
        images = images.permute(0, 3, 1, 2)
        
        return images

class CIFARDataset(Dataset):
    def __init__(self, tokenizer):
        """
        tokenizer: An instance of the PixelTokenizer class with a loaded KMeans model.
        """
        print("Loading CIFAR-10 Data...")
        
        # 1. Define Transforms
        # Note: Ensure these transforms match what your KMeans model was trained on.
        # If KMeans was trained on [0, 1], remove the Normalize step. 
        # If trained on [-1, 1], keep it.
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        ])

        # 2. Load the Raw Dataset
        trainset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )

        # 3. Create a temporary loader to grab all images at once
        # We use a large batch size to load everything into memory efficiently
        temp_loader = DataLoader(trainset, batch_size=50000, shuffle=False)
        
        # Get the big tensor of shape (50000, 3, 32, 32)
        print("Extracting images...")
        all_images, _ = next(iter(temp_loader))

        # 4. Pre-Tokenize Everything
        # This converts (N, 3, 32, 32) -> (N, 1024)
        print("Tokenizing images (this may take a minute)...")
        self.data = tokenizer.tokenize(all_images)
        
        print(f"Loaded and tokenized {len(self.data)} images.")

    def __len__(self):
        # We simply return the total count of images (50,000)
        return len(self.data)

    def __getitem__(self, idx):
        # 1. Retrieve the token sequence for the idx-th image
        # Shape: (1024,)
        tokens = self.data[idx]
        
        # 2. Create Inputs (x) and Targets (y)
        # We are doing "Next Pixel Prediction"
        # x: Tokens 0 to 1022
        # y: Tokens 1 to 1023
        x = tokens[:-1]
        y = tokens[1:]
        
        return x, y