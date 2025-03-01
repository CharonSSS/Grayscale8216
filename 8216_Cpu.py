import tifffile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set device to CPU
device = torch.device('cpu')

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

### Prediction function
def predict_low_bits(input_8bit_path, model_path, output_path):
    # Loading model
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    # Prediction stage
    predict_low_bits(
        input_8bit_path='yourPath/input_8bit_image.tif',
        model_path='unet_model.pth',
        output_path='yourPath/output_16bit_image.tif'
    )