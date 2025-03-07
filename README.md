# DCGAN: Deep Convolutional Generative Adversarial Network

## Overview
This repository contains an implementation of a **Deep Convolutional Generative Adversarial Network (DCGAN)** based on the paper **"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"** by Radford et al. The model is trained to generate realistic images from a given dataset, such as **LSUN Bedrooms** or **CelebA Faces**.

## Dataset Preprocessing
### Supported Datasets:
- **LSUN Bedrooms**
- **CelebA Faces**

### Steps to Preprocess the Dataset:
1. **Download the Dataset:**
   - LSUN: `https://github.com/fyu/lsun`
   - CelebA: `https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html`
2. **Extract and Store Images:** Ensure images are stored in a structured format inside a directory.
3. **Resize Images:** Convert all images to `64x64` pixels using:
   ```python
   from torchvision import transforms
   transform = transforms.Compose([
       transforms.Resize(64),
       transforms.CenterCrop(64),
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))
   ])

## Training the DCGAN Model


### **1. Install Dependencies**
Ensure you have Python 3.8+ and install the required libraries:
```bash
pip install torch torchvision matplotlib numpy tqdm
```

### **2. Train the Model**
Run the training script:
```bash
python train.py --epochs 50 --batch_size 128 --lr 0.0002 --dataset celeba
```
#### Training Options:
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Size of mini-batch (default: 128)
- `--lr`: Learning rate (default: 0.0002)
- `--dataset`: Choose dataset (`celeba` or `lsun`)

The script will output loss values and generate images after each epoch.

### **3. Testing the Model**
To generate new images using a trained model:
```bash
python generate.py --model_path checkpoints/generator.pth
```
This will generate and save synthetic images in the `output/` directory.

## Expected Outputs
During training, the model will:
1. **Initially Generate Random Noise:**
   - Early epochs will produce blurry, unrecognizable shapes.
2. **Gradually Improve Image Quality:**
   - Faces (or bedrooms) will start forming with better structure.
3. **Final Output:**
   - After sufficient training, the model generates realistic images.
   
Example output images:
```
output/
├── generated_epoch_0.png  # Random noise
├── generated_epoch_10.png # Rough shape of faces
├── generated_epoch_50.png # High-quality generated images
```

## References
- [DCGAN Paper](https://arxiv.org/abs/1511.06434)
- [PyTorch DCGAN Example](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
