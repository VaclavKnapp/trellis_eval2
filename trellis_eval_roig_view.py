import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import mediapy
import utils3d

import sys
trellis_path = '/home/vaclav_knapp/TRELLIS'
if trellis_path not in sys.path:
    sys.path.append(trellis_path)

from datasets import load_dataset
from rembg import remove
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils.render_utils import render_frames
from trellis.utils import render_utils, postprocessing_utils

def load_image(image_path, source_size=266):
    """Load and preprocess image for the model."""
    image = torch.from_numpy(np.array(Image.open(image_path)))
    image = image.permute(2,0,1).unsqueeze(0)/255.0 
    if image.shape[1]==4: 
        image = image[:,:3]*image[:,3:] + (1 - image[:,3:])

    h, w = image.shape[2:]
    h_pad = max(0, (w-h)//2)
    w_pad = max(0, (h-w)//2)
    image = F.pad(image, (w_pad,w_pad,h_pad,h_pad), "constant", 1)
    image = F.interpolate(image, size=(source_size, source_size), mode='bicubic', align_corners=True)
    image = torch.clamp(image,0,1)
    return image.squeeze(0).cuda()  

def rotate_pil_image(pil_img, angle):

    return pil_img.rotate(angle, expand=False)

def rotation_matrix_axis_angle(axis, angle):

    ux, uy, uz = axis
    c = torch.cos(angle)
    s = torch.sin(angle)
    one_c = 1.0 - c
    R = torch.tensor([
       [c+ux*ux*one_c,   ux*uy*one_c - uz*s, ux*uz*one_c + uy*s],
       [uy*ux*one_c + uz*s, c+uy*uy*one_c,   uy*uz*one_c - ux*s],
       [uz*ux*one_c - uy*s, uz*uy*one_c + ux*s, c+uz*uz*one_c]
    ], dtype=torch.float32, device='cuda')
    return R

def apply_camera_roll(extr, psi_deg):

    roll_rad = torch.tensor(math.radians(psi_deg + 90))  

    forward_local = torch.tensor([0,0,-1], dtype=torch.float32, device='cuda')
    forward_world = extr[:3,:3] @ forward_local
    forward_world = forward_world / forward_world.norm()


    R_roll = rotation_matrix_axis_angle(forward_world, roll_rad)
    new_extr = extr.clone()
    new_extr[:3,:3] = R_roll @ extr[:3,:3]
    return new_extr

def extrinsics_intrinsics_from_coords(camera_coords, radius=1.7, fov_deg=40.0, device="cuda"):

    extrs = []
    intrs = []
    for (theta_deg, phi_deg, psi_deg) in camera_coords:

        theta = math.radians(theta_deg)
        phi = math.radians(phi_deg)
        

        x = radius * math.sin(theta) * math.cos(phi)
        y = radius * math.sin(theta) * math.sin(phi)
        z = radius * math.cos(theta)
        
        orig = torch.tensor([x, y, z], dtype=torch.float32)


        if abs(theta_deg) < 1e-3 or abs(theta_deg - 180) < 1e-3:
            up_vec = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
        else:
            up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)


        extr = utils3d.torch.extrinsics_look_at(
            orig.cuda(), 
            torch.tensor([0, 0, 0]).float().cuda(), 
            up_vec
        )  


        extr = apply_camera_roll(extr, psi_deg)


        fov_radians = math.radians(fov_deg)
        fov_t = torch.tensor(fov_radians, dtype=torch.float32, device='cuda')
        intr = utils3d.torch.intrinsics_from_fov_xy(fov_t, fov_t)

        extr = extr.to(device)
        intr = intr.to(device)

        extrs.append(extr)
        intrs.append(intr)
    return extrs, intrs

def tensor_to_pil(t: torch.Tensor):

    arr = (t.clamp(0,1)*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(arr)

def main():
    print("Loading MOCHI dataset from HF...")
    mochi_all = load_dataset("tzler/MOCHI")["train"]
    print(f"Total MOCHI trials: {len(mochi_all)}")


    target_idx = None
    for i, ex in enumerate(mochi_all):
        dataset = ex["dataset"]
        trial = ex["trial"]
        if dataset == "barense" and trial == "familiar_high_screen01":
            target_idx = i
            break
    
    if target_idx is None:
        print("Could not find barense/trial 1 in the dataset!")
        return
    
    print(f"Found target trial at index {target_idx}")
    

    print("Initializing Trellis pipeline...")
    os.environ['ATTN_BACKEND'] = 'xformers'
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()
    

    trial_data = mochi_all[target_idx]
    pil_images = trial_data["images"]
    
    if len(pil_images) < 2:
        print(f"Error: Expected at least 2 images in trial, but found {len(pil_images)}")
        return
    

    source_img = pil_images[1]

    rotation_angles = [0, 90, 180, 270]
    rotated_images = [rotate_pil_image(source_img, angle) for angle in rotation_angles]
    

    out_base = Path("data/trellis_rotation_test")
    out_base.mkdir(parents=True, exist_ok=True)
    

    source_img.save(out_base / "original_image.png")
    

    input_dir = out_base / "rotated_inputs"
    input_dir.mkdir(exist_ok=True)
    
    for i, img in enumerate(rotated_images):
        angle = rotation_angles[i]
        img.save(input_dir / f"rotated_{angle}.png")
    

    print("Removing backgrounds from images...")
    processed_dir = out_base / "processed_inputs"
    processed_dir.mkdir(exist_ok=True)
    
    processed_images = []
    for i, img in enumerate(rotated_images):
        angle = rotation_angles[i]
        out_pil = remove(img)
        out_path = processed_dir / f"rotated_{angle}.png"
        out_pil.save(out_path)
        processed_images.append(out_path)
    

    image_tensors = [load_image(str(path)) for path in processed_images]
    

    print("Generating 3D models with TRELLIS...")
    trellis_objects = []
    for i, tensor in enumerate(image_tensors):
        angle = rotation_angles[i]
        print(f"Processing rotation {angle}°...")
        pil_in = transforms.ToPILImage()(tensor)
        out_3d = pipeline.run(pil_in, seed=1)
        gauss = out_3d["gaussian"][0]
        trellis_objects.append(gauss)
    

    print("Rendering 3D models from canonical viewpoint...")

    canonical_view = np.array([[90, 270, 0]], dtype=np.float32)
    
    reconstructed_dir = out_base / "reconstructions"
    reconstructed_dir.mkdir(exist_ok=True)
    
    for i, model in enumerate(trellis_objects):
        angle = rotation_angles[i]
        print(f"Rendering model from {angle}° rotated input...")
        
        e_list, i_list = extrinsics_intrinsics_from_coords(
            canonical_view, 
            radius=1.75,
            fov_deg=40.0
        )
        
        rets = render_frames(
            model,
            extrinsics=e_list,
            intrinsics=i_list,
            options={
                'resolution': 266,
                'bg_color': (1,1,1),
                'near': 0.5,
                'far': 10.0,
                'ssaa': 1
            },
            verbose=False
        )
        
        color_img = rets['color'][0]
        mediapy.write_image(reconstructed_dir / f"from_rotated_{angle}.png", color_img)
    
    print(f"All done! Images saved to {out_base}")

if __name__ == "__main__":
    main()
