import numpy as np
from PIL import Image
import torch
import os 
from tqdm import tqdm 

from unidepth.models import UniDepthV1, UniDepthV2, UniDepthV2old
from unidepth.utils import colorize, image_grid
from unidepth.utils.camera import Pinhole

def extract_number(filename):
    """
    Extract number from filename like 'frame10.png' or 'frame_10.png'
    """
    import re 
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else: 
        return -1 
    

def load_torch_images(folder_path): 
    images = []
    # iterate over all images in the folder sorted by name
    frame_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    sorted_files = []
    if frame_files: 
        sorted_files = sorted(frame_files, key=extract_number)

    for image_name in sorted_files:
        if image_name.endswith(".png"):
            # load image to an array and convert to torch tensor
            rgb = np.array(Image.open(os.path.join(folder_path, image_name)))
            images.append(rgb)
    return images 

def load_ground_truth_depth(folder_path):
    depth_gt = []
    # iterate over all images in the folder sorted by name
    frame_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    sorted_files = []
    if frame_files: 
        sorted_files = sorted(frame_files, key=extract_number)

    for image_name in sorted_files:
        if image_name.endswith(".png"):
            # load depth image to an array and convert like in demo.py
            depth = np.array(Image.open(os.path.join(folder_path, image_name))).astype(float) / 1000.0
            depth_gt.append(depth)
    return depth_gt

def batch_generator(array, batch_size=10):
    """Yield successive batches from frame_paths."""
    for i in range(0, len(array), batch_size):
        yield array[i:i + batch_size]


def demo(model):
    # load camera  
    intrinsics_torch = torch.from_numpy(np.load("assets/demo_2/Intrinsic_L.npy"))
    camera = Pinhole(K=intrinsics_torch.unsqueeze(0))  
    
    # infer method of V1 uses still the K matrix as input
    if isinstance(model, (UniDepthV2old, UniDepthV1)):
        camera = camera.K.squeeze(0)    

    # load images from folder 
    images_folder = "/home/gaps-canteras-u22/Documents/repos/UniDepth/assets/demo_2/Color_L"
    rgb_images = load_torch_images(images_folder)

    ground_truth_folder= "/home/gaps-canteras-u22/Documents/repos/UniDepth/assets/demo_2/Complete_Depth"
    depth_images_gt = load_ground_truth_depth(ground_truth_folder)
    
    os.makedirs("assets/demo_2/outputs", exist_ok=True)

    device = next(model.parameters()).device
    batch_size = 10

    for i in tqdm(range(0, len(rgb_images), batch_size), desc="Processing batches"):
        
        batch_rgb = rgb_images[i:i+batch_size]
        batch_gt = depth_images_gt[i:i+batch_size]

        assert len(rgb_images) == len(depth_images_gt), "Mismatch between RGB and depth frame counts"

        rgb_tensors = torch.stack([
            torch.from_numpy(img).permute(2, 0, 1).float() for img in batch_rgb
        ]).to(device)  # (B, 3, H, W)

        # Camera intrinsics -> No estoy seguro de que esto se tenga que hacer así. 
        if isinstance(model, (UniDepthV2old, UniDepthV1)):
            camera = intrinsics_torch.squeeze(0).to(device)  # V1 expects single matrix
        else:
            camera = Pinhole(K=intrinsics_torch.unsqueeze(0))

        with torch.no_grad():
            predictions = model.infer(rgb_tensors, camera)

        for j, rgb_img in enumerate(batch_rgb):
            depth_pred = predictions["depth"][j].squeeze().cpu().numpy()
            depth_gt = batch_gt[j] # np.array(Image.open(batch_gt[j])).astype(float) / 1000.0

            # Compute relative error
            depth_arel = np.abs(depth_gt - depth_pred) / np.maximum(depth_gt, 1e-6)
            depth_arel[depth_gt == 0.0] = 0.0

            # Colorize
            depth_pred_col = colorize(depth_pred, vmin=0.01, vmax=10.0, cmap="magma_r")
            depth_gt_col = colorize(depth_gt, vmin=0.01, vmax=10.0, cmap="magma_r")
            depth_error_col = colorize(depth_arel, vmin=0.0, vmax=0.2, cmap="coolwarm")

            artifact = image_grid([rgb_img, depth_gt_col, depth_pred_col, depth_error_col], 2, 2)
            out_path = os.path.join("assets/demo_2/outputs", f"frame_{i + j:04d}.png")
            Image.fromarray(artifact).save(out_path)

            print(f"[{i + j:04d}] ARel: {depth_arel[depth_gt > 0].mean() * 100:.2f}%")

if __name__ == "__main__":
    print("Torch version:", torch.__version__)
    type_ = "b"  # available types: s, b, l
    name = f"unidepth-v2-vit{type_}14"
    model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")

    # set resolution level (only V2)
    # model.resolution_level = 9

    # set interpolation mode (only V2)
    model.interpolation_mode = "bilinear"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    demo(model)
