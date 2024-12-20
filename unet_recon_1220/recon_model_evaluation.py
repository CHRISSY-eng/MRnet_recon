import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
import math
from dataset_3view import MRIDataset
from reconstruction_model import UNet3D
import pandas as pd   

# Set device and paths
device_ids = [0, 1, 2]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path = '/home/yaxi/healknee_model/unet_epoch_3043.pth'  # Path to the best model weights
root_dir = "/home/yaxi/MRNet-v1.0_gpu"  # Dataset root directory
visualization_save_path = '/home/yaxi/healknee_model/inference_visualizations'  # Path to save visualizations
labels_files = {
    'abnormal': os.path.join(root_dir, 'valid-abnormal.csv'),
    'acl': os.path.join(root_dir, 'valid-acl.csv'),
    'meniscus': os.path.join(root_dir, 'valid-meniscus.csv')
}

num_visualizations = 10 
# Load the dataset and DataLoader for inference
target_size = (32, 128, 128)
test_dataset_recon = MRIDataset(root_dir, labels_files, phase='valid', target_size=target_size)
test_dataset_recon.labels_abnormal = pd.read_csv(labels_files['abnormal'], header=None, names=['case', 'Label'])
test_dataset_recon.labels_abnormal = test_dataset_recon.labels_abnormal[test_dataset_recon.labels_abnormal['Label'] == 0]
test_loader = DataLoader(test_dataset_recon, batch_size=4, shuffle=False)  # Use batch size of 4 for visualization


unet = UNet3D()

# Load model state_dict and remove the 'module.' prefix if needed
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
unet.load_state_dict(new_state_dict)

# Move the model to the device and wrap in DataParallel if multiple GPUs are available
unet = unet.to(device)
if torch.cuda.device_count() > 1:
    unet = torch.nn.DataParallel(unet, device_ids=device_ids)

unet.eval()

# Define PSNR calculation
def calculate_psnr(true_img, recon_img):
    mse = torch.mean((true_img - recon_img) ** 2).item()
    if mse == 0:
        return 100
    pixel_max = 1.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))

# Define SSIM calculation
def calculate_ssim(true_img, recon_img):
    true_img_np = true_img.cpu().detach().numpy()
    recon_img_np = recon_img.cpu().detach().numpy()
    return ssim(true_img_np, recon_img_np, data_range=true_img_np.max() - true_img_np.min())

# Define MSE calculation
def calculate_mse(true_img, recon_img):
    return torch.mean((true_img - recon_img) ** 2).item()

# Define visualization function
def plot_reconstruction_results(original_images, reconstructed_images, corrupted_boxes, batch_idx, save_path):
    num_images = min(4, original_images.shape[0])  # Plot at most 4 images
    plt.figure(figsize=(12, 8))
    for i in range(num_images):
        # Original image
        plt.subplot(num_images, 2, 2 * i + 1)
        middle_slice_original = original_images[i, 0, original_images.shape[2] // 2, :, :].cpu().detach().numpy()
        plt.imshow(middle_slice_original, cmap='gray')
        plt.title(f'Original Image {i + 1}')
        plt.axis('off')

        # Plot corrupted box on the original image
        box = corrupted_boxes[i]
        rect = plt.Rectangle(
            (box['y'], box['z']),
            box['h'],
            box['w'],
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        plt.gca().add_patch(rect)

        # Reconstructed image
        plt.subplot(num_images, 2, 2 * i + 2)
        middle_slice_reconstructed = reconstructed_images[i, 0, reconstructed_images.shape[2] // 2, :, :].cpu().detach().numpy()
        plt.imshow(middle_slice_reconstructed, cmap='gray')
        plt.title(f'Reconstructed Image {i + 1}')
        plt.axis('off')

    plt.suptitle(f'Reconstruction Results for Batch {batch_idx}')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path)
    plt.close()

# Inference and evaluation function
def perform_inference_and_plot(loader, model, save_dir, num_visualizations):
    psnr_list = []
    ssim_list = []
    mse_list = []
    visualization_count = 0

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device)

            # Create corrupted images
            corrupted_images = images.clone()
            corrupted_boxes = []
            for img_idx, img in enumerate(corrupted_images):
                _, D, H, W = img.shape
                corrupted = False
                max_attempts = 10
                attempts = 0

                # Randomly zero out a region in each image
                while not corrupted and attempts < max_attempts:
                    d, h, w = np.random.randint(D // 4, D // 2), np.random.randint(H // 4, H // 2), np.random.randint(W // 4, W // 2)
                    x = np.random.randint(0, D - d)
                    y = np.random.randint(0, H - h)
                    z = np.random.randint(0, W - w)
                    img[:, x:x + d, y:y + h, z:z + w] = 0
                    corrupted_boxes.append({'x': x, 'y': y, 'z': z, 'd': d, 'h': h, 'w': w})
                    corrupted = True

            # Run model inference
            outputs = model(corrupted_images)

            # Calculate metrics for each image in the batch
            for i in range(images.size(0)):
                psnr = calculate_psnr(images[i], outputs[i])
                ssim_value = calculate_ssim(images[i, 0].cpu(), outputs[i, 0].cpu())
                mse_value = calculate_mse(images[i], outputs[i])

                psnr_list.append(psnr)
                ssim_list.append(ssim_value)
                mse_list.append(mse_value)
            
            if visualization_count < num_visualizations:
                save_path = os.path.join(save_dir, f'inference_visualization_batch_{batch_idx}.png')
                plot_reconstruction_results(images, outputs, corrupted_boxes, batch_idx, save_path)
                visualization_count += 1
            # Visualize the reconstruction results
            
    # Compute average metrics
    avg_psnr = np.mean(psnr_list)
    std_psnr = np.std(psnr_list)

    avg_ssim = np.mean(ssim_list)
    std_ssim = np.std(ssim_list)

    avg_mse = np.mean(mse_list)
    std_mse = np.std(mse_list)

    print(f"Avg PSNR: {avg_psnr:.4f}, Std PSNR: {std_psnr:.4f}")
    print(f"Avg SSIM: {avg_ssim:.4f}, Std SSIM: {std_ssim:.4f}")
    print(f"Avg MSE: {avg_mse:.4f}, Std MSE: {std_mse:.4f}")


# Ensure visualization path exists
if not os.path.exists(visualization_save_path):
    os.makedirs(visualization_save_path)

# Perform inference, evaluation, and save visualizations
perform_inference_and_plot(test_loader, unet, save_dir=visualization_save_path, num_visualizations = num_visualizations)
print(f"Inference and visualizations saved at: {visualization_save_path}")