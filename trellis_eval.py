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

from p3d.losses import calc_l2_losses, calc_lpips_losses

from transformers import (
    AutoImageProcessor, AutoModel,
    CLIPProcessor, CLIPModel
)

RUN_ONLY_10 = False  
EXAMPLE = True

def tensor_to_pil(t: torch.Tensor):

    arr = (t.clamp(0,1)*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(arr)

def rotate(image, degrees):

    if isinstance(image, torch.Tensor):

        if image.dim() == 3 and image.shape[0] == 3:
  
            pil_img = transforms.ToPILImage()(image)
        else:

            pil_img = transforms.ToPILImage()(image.permute(2, 0, 1))
    else:
        pil_img = image
        

    rotated_pil = pil_img.rotate(degrees, fillcolor=(255, 255, 255))
    

    rotated_tensor = transforms.functional.pil_to_tensor(rotated_pil).float() / 255.0
    
    return rotated_tensor

def load_image(image_path, source_size=266):

    image = torch.from_numpy(np.array(Image.open(image_path)))
    image = image.permute(2,0,1).unsqueeze(0)/255.0  # [1,C,H,W]
    if image.shape[1]==4:  # RGBA -> flatten alpha
        image = image[:,:3]*image[:,3:] + (1 - image[:,3:])

    h, w = image.shape[2:]
    h_pad = max(0, (w-h)//2)
    w_pad = max(0, (h-w)//2)
    image = F.pad(image, (w_pad,w_pad,h_pad,h_pad), "constant", 1)
    image = F.interpolate(image, size=(source_size, source_size), mode='bicubic', align_corners=True)
    image = torch.clamp(image,0,1)
    return image.squeeze(0).cuda()  # [3,H,W]

def generate_camera_coords(step=60):

    thetas = np.arange(0, 180+step, step)
    psis = np.arange(0, 360, step)
    camera_coords = []
    for theta_ in thetas:
        num_phis = round(360/step * np.sin(theta_*math.pi/180))
        num_phis = max(num_phis,1)
        phis = np.linspace(0,360,num_phis+1)[:-1]
        for phi_ in phis:
            for psi_ in psis:
                if (theta_==0 or theta_==180) and phi_>0:
                    continue
                camera_coords.append([theta_, phi_, psi_])
    return np.array(camera_coords, dtype=np.float32)

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

    roll_rad = torch.tensor(math.radians(psi_deg))

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
        )  # shape [4,4]

        extr = apply_camera_roll(extr, psi_deg)

        fov_radians = math.radians(fov_deg)
        fov_t = torch.tensor(fov_radians, dtype=torch.float32, device='cuda')
        intr = utils3d.torch.intrinsics_from_fov_xy(fov_t, fov_t)

        extr = extr.to(device)
        intr = intr.to(device)

        extrs.append(extr)
        intrs.append(intr)
    return extrs, intrs

def make_grid_image(matches_tensor):


    n_images = matches_tensor.shape[0]
    height = matches_tensor.shape[3]
    width = matches_tensor.shape[4]
    

    matches_tensor = matches_tensor.permute(0, 1, 3, 4, 2)
    

    grid = torch.zeros((n_images * height, n_images * width, 3))
    

    for i in range(n_images): 
        for j in range(n_images): 

            img = matches_tensor[j, i]  
            

            row_start = i * height
            row_end = (i + 1) * height
            col_start = j * width
            col_end = (j + 1) * width
            

            grid[row_start:row_end, col_start:col_end, :] = img
    
    return grid.cpu().numpy()

def main():
    print("Loading MOCHI dataset from HF...")
    mochi_all = load_dataset("tzler/MOCHI")["train"]
    print(f"Total MOCHI trials: {len(mochi_all)}")

    valid_indices = []
    
    dataset_indices = {
        "barense": [],
        "shapenet": [],
        "shapegen": []
    }
    
    for i, ex in enumerate(mochi_all):
        dataset = ex["dataset"]
        if dataset != "majaj":
            valid_indices.append(i)
            if dataset in dataset_indices:
                dataset_indices[dataset].append(i)
    
    print(f"Excluding 'majaj'; remain {len(valid_indices)} trials.")
    
    if EXAMPLE:
        example_indices = []
        for dataset, indices in dataset_indices.items():
            selected = indices[:10]
            print(f"Selected {len(selected)} examples from {dataset}")
            example_indices.extend(selected)
        
        valid_indices = example_indices
        print(f"Running in EXAMPLE mode with {len(valid_indices)} total trials")

    print("Initializing Trellis pipeline...")
    os.environ['ATTN_BACKEND'] = 'xformers'
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading DINO...")
    dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

    print("Loading CLIP...")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    step = 30
    camera_coords = generate_camera_coords(step=step)
    psis = np.arange(0, 360, step)  
    print(f"Number of camera coords: {len(camera_coords)}")

    n_processed = 0

    for idx in valid_indices:
        i_trial = mochi_all[idx]
        dataset_name = i_trial["dataset"]
        trial_name = i_trial["trial"]
        pil_images = i_trial["images"]  
        n_imgs = len(pil_images)

        print(f"\nProcessing trial index={idx} => {dataset_name}/{trial_name}, #images={n_imgs}")

        tmp_dir = Path(f"tmp_mochi_images/{dataset_name}/{trial_name}")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        trial_paths = []
        for i_img, pil_img in enumerate(pil_images):
            out_pil = remove(pil_img)
            out_path = tmp_dir/f"img_{i_img}.png"
            out_pil.save(out_path)
            trial_paths.append(str(out_path))

        trial_paths.sort()
        trial_imgs = [load_image(pth) for pth in trial_paths]

        trellis_objects = []
        for i_img, t_img in enumerate(trial_imgs):
            pil_in = transforms.ToPILImage()(t_img)
            out_3d = pipeline.run(pil_in, seed=1)
            gauss = out_3d["gaussian"][0]
            trellis_objects.append(gauss)

        brute_force_samples = []
        for j_img, sample_3d in enumerate(trellis_objects):

            sampled_coords = camera_coords[::360//step]
            extr_list, intr_list = extrinsics_intrinsics_from_coords(
                sampled_coords, 
                radius=10,  
                fov_deg=8.0
            )

            rets = render_frames(
                sample_3d,
                extrinsics=extr_list,
                intrinsics=intr_list,
                options={
                    'resolution': 266,
                    'bg_color': (1,1,1), 
                    'near': 0.5,
                    'far': 10.0,
                    'ssaa': 1,
                },
                verbose=False
            )
            color_list = rets['color']

            render_out_folder = Path(f"data/Trellis/{dataset_name}/{trial_name}") / "renders" / f"model_{j_img}"
            render_out_folder.mkdir(parents=True, exist_ok=True)


            all_views = []
            for v_idx, color_arr in enumerate(color_list):

                mediapy.write_image(render_out_folder / f"view_{v_idx:03d}.png", color_arr)
                

                arr_f = color_arr.astype(np.float32)/255.0
                ten = torch.from_numpy(arr_f).permute(2,0,1)  # [C, H, W] format
                

                view_rotations = []
                for psi in psis:
                    rotated_view = rotate(ten, psi)  
                    view_rotations.append(rotated_view)
                

                all_views.extend(view_rotations)
                    
            all_views_tensor = torch.stack(all_views, dim=0) 
            brute_force_samples.append(all_views_tensor)

        brute_force_samples = torch.stack(brute_force_samples, dim=0)


        single_view = np.array([[104.22, 270.0, 0.0]], dtype=np.float32)  
        ref_images_list = []
        for j_img, sample_3d in enumerate(trellis_objects):
            e_list, i_list = extrinsics_intrinsics_from_coords(
                single_view, 
                radius=10,  
                fov_deg=8.0
            )
            rets = render_frames(
                sample_3d,
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
            arr_f = color_img.astype(np.float32)/255.0
            ten = torch.from_numpy(arr_f).permute(2,0,1)
            ref_images_list.append(ten)

        ref_images = torch.stack(ref_images_list, dim=0)

        n_images = len(ref_images)
        _, height, width = ref_images[0].shape


        l2_image_matches = torch.zeros((n_images, n_images, 3, height, width))
        l2_coord_matches = [[None for _ in range(n_images)] for _ in range(n_images)]
        l2_scores = [[None for _ in range(n_images)] for _ in range(n_images)]

        lpips_image_matches = torch.zeros((n_images, n_images, 3, height, width))
        lpips_coord_matches = [[None for _ in range(n_images)] for _ in range(n_images)]
        lpips_scores = [[None for _ in range(n_images)] for _ in range(n_images)]

        clip_image_matches = torch.zeros((n_images, n_images, 3, height, width))
        clip_coord_matches = [[None for _ in range(n_images)] for _ in range(n_images)]
        clip_scores = [[None for _ in range(n_images)] for _ in range(n_images)]

        dino_image_matches = torch.zeros((n_images, n_images, 3, height, width))
        dino_coord_matches = [[None for _ in range(n_images)] for _ in range(n_images)]
        dino_scores = [[None for _ in range(n_images)] for _ in range(n_images)]

        with torch.no_grad():
            for i_img in range(n_images):
                ref_img = ref_images[i_img]


                clip_in = clip_processor(images=tensor_to_pil(ref_img), return_tensors="pt", do_rescale=False)
                clip_in = {k:v.to(device) for k,v in clip_in.items()}
                ref_clip_feat = clip_model.get_image_features(**clip_in).cpu()


                dino_in = dino_processor(images=tensor_to_pil(ref_img), return_tensors="pt", do_rescale=False)
                dino_in = {k:v.to(device) for k,v in dino_in.items()}
                ref_dino_feat = dino_model(**dino_in).pooler_output.cpu()

                for j_img in range(n_images):
                    if i_img == j_img:

                        l2_image_matches[i_img, j_img] = ref_img
                        lpips_image_matches[i_img, j_img] = ref_img
                        clip_image_matches[i_img, j_img] = ref_img
                        dino_image_matches[i_img, j_img] = ref_img

                        l2_coord_matches[i_img][j_img] = np.array([104.22, 270.0, 0.0])
                        lpips_coord_matches[i_img][j_img] = np.array([104.22, 270.0, 0.0])
                        clip_coord_matches[i_img][j_img] = np.array([104.22, 270.0, 0.0])
                        dino_coord_matches[i_img][j_img] = np.array([104.22, 270.0, 0.0])

                        l2_scores[i_img][j_img] = 0.0
                        lpips_scores[i_img][j_img] = 0.0
                        clip_scores[i_img][j_img] = 0.0
                        dino_scores[i_img][j_img] = 0.0
                        continue


                    candidate_views = brute_force_samples[j_img]
                    sampled_coords = camera_coords[::360//step]
                    

                    
                    # L2 comparison
                    l2_vals = calc_l2_losses(ref_img, candidate_views)
                    best_idx = l2_vals.argmin().item()
                    l2_image_matches[i_img, j_img] = candidate_views[best_idx]
                    

                    view_idx = best_idx // len(psis)
                    rotation_idx = best_idx % len(psis)
                    l2_coord_matches[i_img][j_img] = sampled_coords[view_idx].copy()

                    l2_coord_matches[i_img][j_img][2] = psis[rotation_idx]
                    l2_scores[i_img][j_img] = l2_vals[best_idx].item()

                    # LPIPS comparison
                    lpips_vals = calc_lpips_losses(ref_img, candidate_views).flatten()
                    lpips_idx = lpips_vals.argmin().item()
                    lpips_image_matches[i_img, j_img] = candidate_views[lpips_idx]
                    
                    view_idx = lpips_idx // len(psis)
                    rotation_idx = lpips_idx % len(psis)
                    lpips_coord_matches[i_img][j_img] = sampled_coords[view_idx].copy()
                    lpips_coord_matches[i_img][j_img][2] = psis[rotation_idx]
                    lpips_scores[i_img][j_img] = lpips_vals[lpips_idx].item()

                    # CLIP comparison
                    cos_sims = []
                    bs = 16  
                    for start in range(0, len(candidate_views), bs):
                        chunk = candidate_views[start:start+bs]
                        chunk_pil = [tensor_to_pil(c) for c in chunk]
                        c_in = clip_processor(images=chunk_pil, return_tensors="pt", do_rescale=False)
                        c_in = {k:v.to(device) for k,v in c_in.items()}
                        feats = clip_model.get_image_features(**c_in).cpu()
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                        refc = ref_clip_feat / ref_clip_feat.norm(dim=-1, keepdim=True)
                        cos_sims.append((refc @ feats.T).squeeze())
                    cos_sims = torch.cat(cos_sims)
                    clip_losses = 1 - cos_sims
                    clip_idx = clip_losses.argmin().item()
                    clip_image_matches[i_img, j_img] = candidate_views[clip_idx]
                    
                    view_idx = clip_idx // len(psis)
                    rotation_idx = clip_idx % len(psis)
                    clip_coord_matches[i_img][j_img] = sampled_coords[view_idx].copy()
                    clip_coord_matches[i_img][j_img][2] = psis[rotation_idx]
                    clip_scores[i_img][j_img] = clip_losses[clip_idx].item()

                    # DINO comparison
                    dino_sims = []
                    for start in range(0, len(candidate_views), bs):
                        chunk = candidate_views[start:start+bs]
                        chunk_pil = [tensor_to_pil(c) for c in chunk]
                        d_in = dino_processor(images=chunk_pil, return_tensors="pt", do_rescale=False)
                        d_in = {k:v.to(device) for k,v in d_in.items()}
                        feats = dino_model(**d_in).pooler_output.cpu()
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                        refd = ref_dino_feat / ref_dino_feat.norm(dim=-1, keepdim=True)
                        dino_sims.append((refd @ feats.T).squeeze())
                    dino_sims = torch.cat(dino_sims)
                    dino_losses = 1 - dino_sims
                    dino_idx = dino_losses.argmin().item()
                    dino_image_matches[i_img, j_img] = candidate_views[dino_idx]
                    
                    view_idx = dino_idx // len(psis)
                    rotation_idx = dino_idx % len(psis)
                    dino_coord_matches[i_img][j_img] = sampled_coords[view_idx].copy()
                    dino_coord_matches[i_img][j_img][2] = psis[rotation_idx]
                    dino_scores[i_img][j_img] = dino_losses[dino_idx].item()


        l2_coord_matches = np.array(l2_coord_matches)
        lpips_coord_matches = np.array(lpips_coord_matches)
        clip_coord_matches = np.array(clip_coord_matches)
        dino_coord_matches = np.array(dino_coord_matches)

        l2_scores = np.array(l2_scores)
        lpips_scores = np.array(lpips_scores)
        clip_scores = np.array(clip_scores)
        dino_scores = np.array(dino_scores)


        out_path = Path(f"data/Trellis/{dataset_name}/{trial_name}")
        out_path.mkdir(parents=True, exist_ok=True)

        l2_p = out_path/"l2"
        lpips_p = out_path/"lpips"
        clip_p = out_path/"clip"
        dino_p = out_path/"dino"
        
        for p in [l2_p, lpips_p, clip_p, dino_p]:
            p.mkdir(parents=True, exist_ok=True)


        np.save(l2_p/"coords.npy", l2_coord_matches)
        np.save(l2_p/"losses.npy", l2_scores)
        np.save(lpips_p/"coords.npy", lpips_coord_matches)
        np.save(lpips_p/"losses.npy", lpips_scores)
        np.save(clip_p/"coords.npy", clip_coord_matches)
        np.save(clip_p/"losses.npy", clip_scores)
        np.save(dino_p/"coords.npy", dino_coord_matches)
        np.save(dino_p/"losses.npy", dino_scores)


        for i_img in range(n_images):
            for j_img in range(n_images):
                l2_np = l2_image_matches[i_img, j_img].permute(1, 2, 0).cpu().numpy()
                mediapy.write_image(l2_p/f"viewpoint{i_img}_model{j_img}.png", l2_np)

                lpips_np = lpips_image_matches[i_img, j_img].permute(1, 2, 0).cpu().numpy()
                mediapy.write_image(lpips_p/f"viewpoint{i_img}_model{j_img}.png", lpips_np)

                clip_np = clip_image_matches[i_img, j_img].permute(1, 2, 0).cpu().numpy()
                mediapy.write_image(clip_p/f"viewpoint{i_img}_model{j_img}.png", clip_np)

                dino_np = dino_image_matches[i_img, j_img].permute(1, 2, 0).cpu().numpy()
                mediapy.write_image(dino_p/f"viewpoint{i_img}_model{j_img}.png", dino_np)


        mediapy.write_image(l2_p/"grid.png", make_grid_image(l2_image_matches))
        mediapy.write_image(lpips_p/"grid.png", make_grid_image(lpips_image_matches))
        mediapy.write_image(clip_p/"grid.png", make_grid_image(clip_image_matches))
        mediapy.write_image(dino_p/"grid.png", make_grid_image(dino_image_matches))


        orig_view = out_path/"original_viewpoints"
        orig_view.mkdir(parents=True, exist_ok=True)
        for i_img, ref_img in enumerate(ref_images):
            ref_np = ref_img.permute(1, 2, 0).cpu().numpy()
            mediapy.write_image(orig_view/f"ref_{i_img}.png", ref_np)
        n_processed += 1

    print(f"All done! Processed {n_processed} trials.")

if __name__ == "__main__":
    main()
