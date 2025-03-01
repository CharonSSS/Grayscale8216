# Grayscale8216
Machine learning raising 8-bit grayscale heightmap up to 16-bit, without terraces and precision loses compared to the original 8-bit heightmap. Written by AI and [Aebestach](https://github.com/Aebestach).

## Demonstration and Comparison:

### From top to bottom are (Example of Mimas, which has original 16-bit)
* 8-bit heightmap derived from original 16-bit
* 16-bit heightmap converted from the 8-bit above using machine learing
* original 16-bit

![image](https://github.com/user-attachments/assets/b36db404-1b61-4330-8f2d-4b8513678122)
![image](https://github.com/user-attachments/assets/541ed1de-e6c3-47ae-9276-673606159f0d)
![image](https://github.com/user-attachments/assets/c41c61d9-e66f-4853-b627-2962e4492a94)

### From top to bottom are (Example of Io, which doesn't have original 16-bit)
* original 8-bit heightmap
* 16-bit heightmap converted from the 8-bit above using machine learing

![image](https://github.com/user-attachments/assets/d908fe6b-5dc9-45d1-8589-9350921c65ad)
![image](https://github.com/user-attachments/assets/f3f311a2-8921-457c-8217-5412eb80f693)

## Usage

### If you only want to upgrade your 8-bit images to 16-bit
* Clone or download the repo to your local
* Install Python
* Execute pip install numpy, pip install tifffile, pip install torch
* Open 8216_Cpu.py, set up the file paths and run the script **(Please note that this process largely depends on your RAM. If there is a out of memory error, divide the image into small pieces, and then stitch them together)**
* [For who wants to run it on GPU] Execute pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
* [For who wants to run it on GPU] Open 8216_AutoCpuGpu.py, set up the file paths and run the script **(Please note that this process largely depends on your Video Memory. If there is a out of memory error, divide the image into small pieces, and then stitch them together)**

### If you want to train the model yourself
* Clone or download the repo to your local
* Install Python
* Execute pip install numpy, pip install tifffile, pip install torch
* Open modelTraining_Cpu.py, set up the dataset path and run the script
* [For who wants to run it on GPU] Execute `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`
* [For who wants to run it on GPU] Open modelTraining_AutoCpuGpu.py, set up the dataset path and run the script
