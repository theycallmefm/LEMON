import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from PIL import Image
from controlnet import MultiControlNet
import trimesh
import json
import os

def generate_heatmap_parallel(images, text: str, processor, model) -> torch.Tensor:
    """
    Generate heatmaps for a batch of views using the provided text prompt.
    
    Args:
        images (List[image]): A list of image objects containing color images 255.
        text (str): The text prompt for generating the segmentation masks.
        processor: The CLIPSeg processor.
        model: The CLIPSeg model.
        
    Returns:
        torch.Tensor: A batch of heatmaps with shape (B, 1, H, W).
    """
    # Extract color images from views and collate them into a batch tensor
    
    
    # Check if images have an alpha channel (C == 4) and remove it if present
    if images.shape[-1] == 4:
        images = images[..., :3]

    # Ensure images are in the range [0, 1] and in the correct shape (B, C, H, W)
    images = images.permute(0, 3, 1, 2)  # Change shape to (B, C, H, W)
    
    # Preprocess the images and text
    inputs = processor(images = images, text=[text] * images.size(0), return_tensors="pt", padding=True).to(images.device)
    
    # Generate the segmentation masks
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    masks = logits.sigmoid()  # shape (B, 1, 352, 352)
    
    # Ensure masks have the shape (B, 1, H, W)
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)  # Ensure it's in (B, 1, H, W)
    
    # Resize the masks to match the original image dimensions
    original_size = images.shape[2:4]  # (height, width)
    masks_resized = F.interpolate(masks, size=original_size, mode='bilinear', align_corners=False)  # shape (B, 1, H, W)
    # Normalize the mask values to be between 0 and 1 for each image in the batch
    min_val = masks_resized.view(masks_resized.size(0), -1).min(dim=1, keepdim=True)[0].view(masks_resized.size(0), 1, 1, 1)
    max_val = masks_resized.view(masks_resized.size(0), -1).max(dim=1, keepdim=True)[0].view(masks_resized.size(0), 1, 1, 1)
    masks_normalized = (masks_resized - min_val) / (max_val - min_val + 1e-8)
    
    return masks_normalized  # shape (B, 1, H, W)

def save_heatmap_as_png(heatmap_tensor: torch.Tensor, filename: str):
    # Convert the tensor to a NumPy array
    heatmap_np = heatmap_tensor.squeeze().cpu().numpy()
    
    # Use a colormap to create a color image
    cmap = plt.get_cmap('jet')
    heatmap_color = cmap(heatmap_np)
    
    # Convert the heatmap color values to 8-bit integers
    heatmap_color = (heatmap_color[:, :, :3] * 255).astype(np.uint8)
    
    # Save the color heatmap as a PNG file
    plt.imsave(filename, heatmap_color)
    return heatmap_color

def save_colored_mesh(mesh, vertex_colors, filename):
    # Create a trimesh object with vertex colors
    tri_mesh = trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(),
                               faces=mesh.indices.detach().cpu().numpy(),
                               vertex_colors=vertex_colors.detach().cpu().numpy())
    
    # Export the mesh to a file
    tri_mesh.export(filename)

def depth_filter(gbuffer, max_depth_diff=100000):
    depth = gbuffer["depth"]
    valid_mask = (depth >= 0.0) & (depth <= max_depth_diff)
    return valid_mask

def match_faces_to_pixels_batch(gbuffer_batch):
    """
    Matches mesh faces to their corresponding pixels for a batch of G-buffers.

    Args:
        gbuffer_batch (list): List of G-buffers containing rasterized outputs.

    Returns:
        list: List of dictionaries where keys are face indices and values are lists of pixel coordinates.
    """
    face_to_pixels_batch = []

    for gbuffer in gbuffer_batch:
        H, W, _ = gbuffer["position"].shape
        face_idx = gbuffer["face_idx"].squeeze(0).squeeze(-1)  # Shape: [H, W]
        face_to_pixels = {}

        for y in range(H):
            for x in range(W):
                f_id = face_idx[y, x].item()
                if f_id >= 0:  # Only consider valid face indices
                    if f_id not in face_to_pixels:
                        face_to_pixels[f_id] = []
                    face_to_pixels[f_id].append((y, x))

        face_to_pixels_batch.append(face_to_pixels)
    
    return face_to_pixels_batch

def color_mesh_from_image_batch(face_to_pixels_batch, mesh, heatmaps, depth_filters):
    """
    Assign colors to mesh vertices based on the corresponding pixels in the image for a batch of images.

    Args:
        face_to_pixels_batch (list): List of dictionaries mapping face indices to pixel coordinates.
        mesh (trimesh.Trimesh): Mesh object.
        images (torch.Tensor): Batch of image tensors of shape (B, H, W, C).
        heatmaps (torch.Tensor): Batch of heatmaps of shape (B, H, W, 1).
        depth_filters (list): List of depth filters for each image in the batch.

    Returns:
        torch.Tensor: Vertex scores of shape (N, 1).
        torch.Tensor: Vertex colors of shape (N, C).
    """
    vertex_scores = torch.zeros((mesh.vertices.shape[0], heatmaps.shape[3]), device=heatmaps.device)
    vertex_count = torch.zeros((mesh.vertices.shape[0]), device=heatmaps.device)

    max_faces = mesh.indices.shape[0]

    for i, face_to_pixels in enumerate(face_to_pixels_batch):
        print("calculating view ",i)
        heatmap = heatmaps[i]
        depth_filter = depth_filters[i]
        
        for face_idx, pixels in face_to_pixels.items():
            if face_idx >= max_faces:  # Check indexing
                print(f"Skipping face_idx {face_idx} as it exceeds max_faces {max_faces}")
                continue
            # valid_pixels = [p for p in pixels if depth_filter[p[0], p[1]]]
            # if valid_pixels:
            pixel_scores = torch.stack([heatmap[y, x] for y, x in pixels])
            #mean_color = pixel_colors.mean(dim=0)
            mean_score = pixel_scores.mean(dim=0)

            for vertex_idx in mesh.indices[face_idx]:
                vertex_scores[vertex_idx] += mean_score
                #vertex_colors[vertex_idx] += mean_color
                vertex_count[vertex_idx] += 1

    nonzero_mask = vertex_count > 0
    vertex_scores[nonzero_mask] /= vertex_count[nonzero_mask].unsqueeze(-1)

    return vertex_scores

def depth_filter_batch(gbuffer_batch, max_depth_diff=100000):
    """
    Generates depth filters for a batch of G-buffers.

    Args:
        gbuffer_batch (list): List of G-buffers containing depth information.
        max_depth_diff (float): Maximum allowable depth difference.

    Returns:
        list: List of valid depth masks for each G-buffer.
    """
    depth_filters = []
    for gbuffer in gbuffer_batch:
        depth = gbuffer["depth"]
        valid_mask = (depth >= 0.0) & (depth <= max_depth_diff)
        depth_filters.append(valid_mask)
    
    return depth_filters

def map_pixels_to_vertices(gbuffers, mesh, heatmaps, images, depth_filters):
    """
    Map each pixel value and score with mesh indices.
    
    Args:
        gbuffers (list): List of G-buffers containing rasterized outputs.
        mesh (trimesh.Trimesh): Mesh object.
        heatmaps (torch.Tensor): Batch of heatmaps of shape (B, H, W, 1).
        images (torch.Tensor): Batch of images with shape (B, H, W, C).
        depth_filters (torch.Tensor): Batch of depth filters for each image in the batch.
        
    Returns:
        torch.Tensor: Vertex scores of shape (N, 1).
        torch.Tensor: Vertex colors of shape (N, C).
    """
    B, H, W, C = images.shape
    V = mesh.vertices.shape[0]
    vertex_scores = torch.zeros((V, 1), device=images.device, dtype=torch.float32)
    vertex_colors = torch.zeros((V, C), device=images.device, dtype=torch.float32)
    vertex_count = torch.zeros((V), device=images.device, dtype=torch.float32)

    for b in range(B):
        face_idx = gbuffers[b]["face_idx"].squeeze(0).squeeze(-1)  # Shape: [H, W]
        valid_pixels = depth_filters[b].squeeze(-1)  # Shape: [H, W]
        
        for y in range(H):
            for x in range(W):
                if valid_pixels[y, x]:
                    f_id = face_idx[y, x].item()
                    if f_id >= 0:
                        pixel_color = images[b, y, x]
                        pixel_score = heatmaps[b, y, x]
                        
                        for vertex_idx in mesh.indices[f_id]:
                            vertex_scores[vertex_idx] += pixel_score
                            vertex_colors[vertex_idx] += pixel_color
                            vertex_count[vertex_idx] += 1

    nonzero_mask = vertex_count > 0
    vertex_colors[nonzero_mask] /= vertex_count[nonzero_mask].unsqueeze(-1)
    vertex_scores[nonzero_mask] /= vertex_count[nonzero_mask].unsqueeze(-1)

    return vertex_scores, vertex_colors
        
def load_scores_from_json(file_path, device):
    with open(file_path, 'r') as f:
        data = json.load(f)
    prompt = data["text_prompt"]
    vertex_scores = torch.tensor(data["vertex_scores"], device=device)
    normalized_scores = [torch.tensor(view["normalized_scores"],device=device) for view in data["views"]]
    return prompt, normalized_scores, vertex_scores

def save_scores_to_json(prompt, normalized_scores, vertex_scores, file_path):
    """
    Save rendered scores and vertex scores to a JSON file.

    Args:
        normalized_scores (list): List of normalized score arrays for each view.
        vertex_scores (torch.Tensor): Vertex scores.
        file_path (str): Path to the JSON file.
    """
    data = {
        "text_prompt" : prompt,
        "views": [],
        "vertex_scores": vertex_scores.cpu().numpy().tolist()
    }

    for i, score in enumerate(normalized_scores):
        view_data = {
            "view_index": i,
            "normalized_scores": score.cpu().numpy().tolist()
        }
        data["views"].append(view_data)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def paint_mesh_parallel(views, mesh, renderer, text, processor, model, json_path="", mesh_heatmaps_path="", heatmaps_path=""):
    #json_file = os.path.join(json_path, text.replace(" ", "_") + "_rendered_scores.json")
    if os.path.exists(json_path):
        print("Loading scores and heatmaps from JSON file.")
        _, scores, vertex_scores = load_scores_from_json(json_path, device=views[0].device)
        for i, score in enumerate(scores):
            views[i].heatmap = score
            if os.path.exists(mesh_heatmaps_path):
                save_heatmap_as_png(heatmap_tensor=score, filename=str(mesh_heatmaps_path) + "/" + str(i) + "_mesh_heatmap.png")
    
    else:
        with torch.no_grad():
            print("rendering")
            gbuffers = renderer.render(views, mesh, channels=['position', 'depth'], with_antialiasing=True)
            images = torch.stack([view.color * 255 for view in views]).to(views[0].device)
            print("heatmap generating")
            heatmaps = generate_heatmap_parallel(images, text, processor, model)

            # Update each view with its heatmap
            print("coloring")
            for i in range(len(views)):
                #views[i].heatmap = heatmaps[i]
                if os.path.exists(heatmaps_path):
                    color = save_heatmap_as_png(heatmap_tensor=heatmaps[i], filename=str(heatmaps_path) + "/" + str(i) + "_heatmap.png")
                color = torch.from_numpy(color).to(views[i].device)

            heatmaps = rearrange(heatmaps, "b c h w -> b h w c")
            #print(heatmaps[0].min(), heatmaps[0].max())
            # Parallel processing
            print("depth_filters")
            depth_filters = depth_filter_batch(gbuffers)
            print("match_faces_to_pixels_batch")
            face_to_pixels_batch = match_faces_to_pixels_batch(gbuffers)
            print("color_mesh_from_image_batch")
            vertex_scores = color_mesh_from_image_batch(face_to_pixels_batch, mesh, heatmaps, depth_filters)
            vertex_scores = (vertex_scores - vertex_scores.min()) / (vertex_scores.max() - vertex_scores.min())
            
            

            gbuffers = renderer.render(views, mesh, channels=['score'], with_antialiasing=True, vertex_scores=vertex_scores)
            scores = [gbuffer["score"] for gbuffer in gbuffers]  # List of score arrays, one per view

            #normalized_scores = [(score - score.min()) / (score.max() - score.min()) for score in scores] #TODO check if it is necessary
            for i, new_heatmap in enumerate(scores):
                views[i].heatmap = new_heatmap
                if os.path.exists(mesh_heatmaps_path):
                    save_heatmap_as_png(heatmap_tensor = new_heatmap, filename=str(mesh_heatmaps_path) + "/" + str(i) + "_mesh_heatmap.png")
        
            save_scores_to_json(text, scores, vertex_scores, json_path)

    
    vertex_colors = vertex_scores.squeeze().cpu().numpy()
    cmap = plt.get_cmap('jet')
    vertex_colors = cmap(vertex_colors)
    vertex_colors = (vertex_colors[:, :3] * 255)
    vertex_colors = torch.from_numpy(vertex_colors).to(views[0].device)
    
    return views, vertex_scores, vertex_colors

def mask_image(heatmap, generated_img, ground_truth_img, threshold):
    """
    Apply a mask based on the heatmap and threshold to replace corresponding pixels in the ground truth image with those from the generated image.

    Args:
        heatmap (torch.Tensor): Heatmap of shape (1, H, W).
        generated_img (torch.Tensor): Generated image of shape (H, W, C).
        ground_truth_img (torch.Tensor): Ground truth image of shape (H, W, C).
        threshold (float): Threshold value to create the mask.

    Returns:
        torch.Tensor: Modified ground truth image with masked values from the generated image.
    """
    # Create the mask based on the threshold
    mask = heatmap >= threshold  # Shape: (1, H, W)

    # Squeeze the mask to remove the first dimension
    mask_squeezed = mask.squeeze(2)  # Shape: (H, W)

    # Expand the mask to match the shape of the images
    mask_expanded = mask_squeezed.unsqueeze(-1).expand_as(generated_img)  # Shape: (H, W, C)

    # Apply the mask to replace values in the ground truth image with those from the generated image
    modified_ground_truth_img = torch.where(mask_expanded, generated_img, ground_truth_img)

    return modified_ground_truth_img, mask

def blend_images_with_alpha(heatmap, generated_img, input_img, threshold):
    """
    Blend the generated image with the input image using the heatmap as an alpha channel, keeping values above a threshold.

    Args:
        heatmap (torch.Tensor): Heatmap tensor with shape (1, H, W) and values between 0 and 1.
        generated_img (torch.Tensor): Generated image tensor with shape (H, W, 3).
        input_img (torch.Tensor): Input image tensor with shape (H, W, 3).
        threshold (float): Threshold value to mask the heatmap.

    Returns:
        torch.Tensor: Blended image tensor with shape (H, W, 3).
    """
    # Apply the threshold to the heatmap
    mask = heatmap >= threshold  # Shape: (1, H, W)
    heatmap = heatmap * mask.float()

    # Ensure the heatmap values are between 0 and 1
    heatmap = torch.clamp(heatmap, 0, 1)

    # Remove any extra dimensions from the heatmap
    heatmap = heatmap.squeeze()
    if len(heatmap.shape) == 2:
        heatmap = heatmap.unsqueeze(-1)

    # Expand the heatmap to match the shape of the images
    alpha = heatmap.expand_as(generated_img)

    # Blend the images using the alpha channel
    blended_img = generated_img * alpha + input_img * (1 - alpha)
    
    return blended_img

