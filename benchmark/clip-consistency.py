from transformers import (
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
from PIL import Image
from argparse import ArgumentParser
from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda")

clip_id = "openai/clip-vit-large-patch14"
image_processor = CLIPImageProcessor.from_pretrained(clip_id)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)


class DirectionalConsistency(nn.Module):
    def __init__(self, image_processor, image_encoder):
        super().__init__()
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    def preprocess_image(self, image):
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.to(device)}

    def encode_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def compute_directional_consistency(self, original_0, original_1, edited_0, edited_1):
        sim_direction = F.cosine_similarity(original_1 - original_0, edited_1 - edited_0)
        return sim_direction

    def forward(self, original_0, original_1, edited_0, edited_1):
        feat_original_2 = self.encode_image(original_0)
        feat_original_1 = self.encode_image(original_1)
        feat_edited_0 = self.encode_image(edited_0)
        feat_edited_1 = self.encode_image(edited_1)
        directional_consistency = self.compute_directional_consistency(
            feat_original_2, feat_original_1, feat_edited_0, feat_edited_1
        )
        return directional_consistency

def get_file_names(directory, extension):
    """Fetch all file names with a specific extension from the given directory."""
    return [file for file in os.listdir(directory) if file.endswith(extension) and os.path.isfile(os.path.join(directory, file))]

def read_images(images_dir, extension="png"):
    file_names = get_file_names(images_dir, extension)
    # Sort the file names alphabetically
    file_names.sort()
    image_paths = [os.path.join(images_dir, file_name) for file_name in file_names]
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    images = [np.array(image) for image in images]
    return images


def plot_consistency(args):
    input_views = read_images(images_dir=args.input_dir, extension=args.pic_extension)
    dir_consistency = DirectionalConsistency(image_processor, image_encoder)
    saved_iterations = range(args.start_iter, args.end_iter+1, 1000)
    iteration_scores = []
    for iteration in saved_iterations:
        step_dir = args.edit_dir / str(iteration)
        edited_views = read_images(images_dir=step_dir, extension=args.pic_extension)
        scores = []
        for i in range(len(input_views)-1):
            original_image_0 = input_views[i]
            original_image_1 = input_views[i+1]
            edited_image_0 = edited_views[i]
            edited_image_1 = edited_views[i+1]
            similarity_score = dir_consistency(
                original_image_0, original_image_1, edited_image_0, edited_image_1
            )
            scores.append(float(similarity_score.detach().cpu()))
        iteration_mean = np.mean(scores)
        iteration_scores.append(np.mean(scores))
        print(f"Mean CLIP directional consistency at step {iteration}: {iteration_mean}")
    plt.plot(saved_iterations, iteration_scores, 'o-')
    plt.xlabel('Iteration')
    plt.ylabel('Mean CLIP Directional Consistency Score')
    plt.title(args.fig_title)
    plt.savefig(args.edit_dir / "mean_clip_consistency.png")

def compute_consistency(args):
    input_views = read_images(images_dir=args.input_dir, extension=args.pic_extension)
    edited_views = read_images(images_dir=args.edit_dir, extension=args.pic_extension)
    dir_consistency = DirectionalConsistency(image_processor, image_encoder)
    scores = []
    for i in range(len(input_views)-1):
        original_image_0 = input_views[i]
        original_image_1 = input_views[i+1]
        edited_image_0 = edited_views[i]
        edited_image_1 = edited_views[i+1]
        similarity_score = dir_consistency(
            original_image_0, original_image_1, edited_image_0, edited_image_1
        )
        scores.append(float(similarity_score.detach().cpu()))
    mean_score = np.mean(scores)
    print(f"Mean CLIP directional consistency: {mean_score}")
    return


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=str)
    parser.add_argument("--edit-dir", required=True, type=str)
    parser.add_argument("--pic-extension", default="png", type=str)
    parser.add_argument("--start-iter", default=2000, type=int)
    parser.add_argument("--end-iter", default=10000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--fig-title", default="", type=str)
    args = parser.parse_args()
    args.input_dir = Path(args.input_dir)
    args.edit_dir = Path(args.edit_dir)
    torch.manual_seed(args.seed)
    plot_consistency(args)

if __name__=="__main__":
    main()
