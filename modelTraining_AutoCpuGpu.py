import tifffile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import os
from tqdm import tqdm

# Set up the device (use GPU if available, use CPU otherwise)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Define U-Net model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Intermediate layer
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)

        # Decoder
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4_1 = nn.Conv2d(256 + 128, 128, 3, padding=1)
        self.conv4_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5_1 = nn.Conv2d(128 + 64, 64, 3, padding=1)
        self.conv5_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.out_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.conv1_1(x))
        x1 = F.relu(self.conv1_2(x1))
        p1 = self.pool1(x1)

        x2 = F.relu(self.conv2_1(p1))
        x2 = F.relu(self.conv2_2(x2))
        p2 = self.pool2(x2)

        # Intermediate layer
        x3 = F.relu(self.conv3_1(p2))
        x3 = F.relu(self.conv3_2(x3))

        # Decoder
        u2 = self.up2(x3)
        concat2 = torch.cat([x2, u2], dim=1)  # Splicing along the channel dimension
        x4 = F.relu(self.conv4_1(concat2))
        x4 = F.relu(self.conv4_2(x4))

        u1 = self.up1(x4)
        concat1 = torch.cat([x1, u1], dim=1)  # Splicing along the channel dimension
        x5 = F.relu(self.conv5_1(concat1))
        x5 = F.relu(self.conv5_2(x5))

        out = torch.sigmoid(self.out_conv(x5))
        return out

### Custom TIFF Image Dataset
class TiffDataset(Dataset):
    def __init__(self, image_paths, patch_size=256):
        self.image_paths = image_paths
        self.patch_size = patch_size

    def __len__(self):
        # Fixed length, for simplicity, can be adjusted as needed
        return 10000

    def __getitem__(self, idx):
        # Randomly select an image
        path = np.random.choice(self.image_paths)
        image = tifffile.imread(path).astype(np.uint16)

        # Extract and standardize the high 8 bits and low 8 bits
        high = (image >> 8).astype(np.float32) / 255.0
        low = (image & 0xFF).astype(np.float32) / 255.0

        # Random cropping
        h, w = high.shape
        i = np.random.randint(0, h - self.patch_size)
        j = np.random.randint(0, w - self.patch_size)

        high_patch = high[i:i+self.patch_size, j:j+self.patch_size]
        low_patch = low[i:i+self.patch_size, j:j+self.patch_size]

        # Add channel dimension (1, H, W)
        return high_patch[np.newaxis, ...], low_patch[np.newaxis, ...]

### Training function（Add tqdm progress bar）
def train_model(train_data_folder, num_epochs=10, batch_size=16, model_save_path='unet_model.pth', checkpoint_path='checkpoint.pth'):
    # Retrieve all. tif files in the folder
    train_images = glob.glob(os.path.join(train_data_folder, '*.tif'))
    if not train_images:
        raise ValueError(f"No .tif files were found in the folder {train_data_folder}!")

    # Initialize model
    model = UNet()
    if torch.cuda.device_count() > 1:
        print(f"Train using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Initialize training parameters
    start_epoch = 0
    prev_loss = float('inf')

    # Load checkpoint (if present)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Starting from the next epoch
        prev_loss = checkpoint['loss']
        print(f"Continue training from epoch {start_epoch}, previous loss: {prev_loss}")

    model.train()

    # Create data loader
    dataset = TiffDataset(train_images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Circuit training
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        # Display progress bar using tqdm
        with tqdm(total=100, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for step, (high, low) in enumerate(dataloader):
                if step >= 100:  # Step limit per epoch
                    break
                high = high.float().to(device)
                low = low.float().to(device)

                optimizer.zero_grad()
                output = model(high)
                loss = criterion(output, low)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())  # Display the loss of the current batch
                pbar.update(1)  # Update progress bar

        # Calculate the average loss of the current epoch
        avg_loss = epoch_loss / min(100, len(dataloader))
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}')

        # If the current loss is smaller than the loss of the previous epoch, save the model
        if avg_loss < prev_loss:
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), model_save_path)
            else:
                torch.save(model.state_dict(), model_save_path)
            print(f"The model has been saved, and the current loss {avg_loss} is less than the previous loss {prev_loss}")

        # Update prev_Loss to current loss
        prev_loss = avg_loss

        # Saving Checkpoints
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)

    return model

### Prediction function
def predict_low_bits(input_8bit_path, model_path, output_path):
    # Loading model
    model = UNet()
    if torch.cuda.device_count() > 1:
        print(f"Train using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Read and standardize the input 8-bit image
    input_8bit = tifffile.imread(input_8bit_path).astype(np.uint8)
    input_norm = input_8bit.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(input_norm[np.newaxis, np.newaxis, ...]).float().to(device)

    # Predict low 8 bits
    with torch.no_grad():
        predicted_low = model(input_tensor)
    predicted_low = predicted_low.squeeze().cpu().numpy()
    predicted_low = (predicted_low * 255).clip(0, 255).astype(np.uint8)

    # Merge into a 16 bit image
    output_16bit = (input_8bit.astype(np.uint16) << 8) | predicted_low

    # Save the Results
    tifffile.imwrite(output_path, output_16bit, dtype=np.uint16)

### Example usage
if __name__ == "__main__":
    # Training stage
    train_data_folder = 'your16bitDatasets'
    train_model(train_data_folder, num_epochs=100, batch_size=16)

    # Prediction stage
    #predict_low_bits(
    #    input_8bit_path='yourPath/input_8bit_image.tif',
    #    model_path='unet_model.pth',
    #    output_path='yourPath/output_16bit_image.tif'
    #)