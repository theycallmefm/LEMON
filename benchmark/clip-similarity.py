from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
# from einops import rearrange
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
tokenizer = CLIPTokenizer.from_pretrained(clip_id)
text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(device)
image_processor = CLIPImageProcessor.from_pretrained(clip_id)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)


class DirectionalSimilarity(nn.Module):
    def __init__(self, tokenizer, text_encoder, image_processor, image_encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.image_encoder = image_encoder
        # name = "ViT-L/14"
        # self.size = {"RN50x4": 288, "RN50x16": 384, "RN50x64": 448, "ViT-L/14@336px": 336}.get(name, 224)
        # self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        # self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def preprocess_image(self, image):
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.to(device)}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.to(device)}

    def encode_image(self, image):
        # image = F.interpolate(image.float(), size=self.size, mode="bicubic", align_corners=False)
        # image = image - rearrange(self.mean, "c -> 1 c 1 1")
        # image = image / rearrange(self.std, "c -> 1 c 1 1")
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def encode_text(self, text):
        tokenized_text = self.tokenize_text(text)
        text_features = self.text_encoder(**tokenized_text).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute_directional_similarity(self, img_feat_one, img_feat_two, text_feat_one, text_feat_two):
        sim_direction = F.cosine_similarity(img_feat_two - img_feat_one, text_feat_two - text_feat_one)
        return sim_direction

    def forward(self, image_one, image_two, caption_one, caption_two):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        text_feat_one = self.encode_text(caption_one)
        text_feat_two = self.encode_text(caption_two)
        directional_similarity = self.compute_directional_similarity(
            img_feat_one, img_feat_two, text_feat_one, text_feat_two
        )
        return directional_similarity


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

def iteration_similarity_plot(args):
    args.input_dir = Path(args.input_dir)
    args.edit_dir = Path(args.edit_dir)
    torch.manual_seed(args.seed)
    input_views = read_images(images_dir=args.input_dir, extension=args.pic_extension)
    dir_similarity = DirectionalSimilarity(tokenizer, text_encoder, image_processor, image_encoder)
    # if args.start_iter == None or args.end_iter == None:
    #     edited_views = read_images(images_dir=args.edit_dir, extension=args.pic_extension)
    #     scores = []
    #     for i in range(len(input_views)):
    #         original_image = input_views[i]
    #         edited_image = edited_views[i]
    #         similarity_score = dir_similarity(
    #             original_image, edited_image, args.original_caption, args.edited_caption
    #         )
    #         scores.append(float(similarity_score.detach().cpu()))
    #     mean_score = np.mean(scores)
    #     print(f"Mean CLIP directional similarity: {mean_score}")
    #     return
    saved_iterations = range(args.start_iter, args.end_iter+1, 1000)
    iteration_scores = []
    for iteration in saved_iterations:
        step_dir = args.edit_dir / str(iteration)
        edited_views = read_images(images_dir=step_dir, extension=args.pic_extension)
        scores = []
        for i in range(len(input_views)):
            original_image = input_views[i]
            edited_image = edited_views[i]
            similarity_score = dir_similarity(
                original_image, edited_image, args.original_caption, args.edited_caption
            )
            scores.append(float(similarity_score.detach().cpu()))
        iteration_mean = np.mean(scores)
        iteration_scores.append(np.mean(scores))
        print(f"Mean CLIP directional similarity at step {iteration}: {iteration_mean}")
    plt.plot(saved_iterations, iteration_scores, 'o-')
    plt.xlabel('Iteration')
    plt.ylabel('Mean CLIP Directional Similarity Score')
    plt.title(args.fig_title)
    plt.savefig(args.edit_dir / "mean_clip_directional.png")

def compute_mean_clip_similarity(args):
    args.input_dir = Path(args.input_dir)
    args.edit_dir = Path(args.edit_dir)
    torch.manual_seed(args.seed)
    input_views = read_images(images_dir=args.input_dir, extension=args.pic_extension)
    edited_views = read_images(images_dir=args.edit_dir, extension=args.pic_extension)
    dir_similarity = DirectionalSimilarity(tokenizer, text_encoder, image_processor, image_encoder)
    scores = []
    for i in range(len(input_views)):
        original_image = input_views[i]
        edited_image = edited_views[i]
        similarity_score = dir_similarity(
            original_image, edited_image, args.original_caption, args.edited_caption
        )
        scores.append(float(similarity_score.detach().cpu()))
    mean_score = np.mean(scores)
    print(f"Mean CLIP directional similarity: {mean_score}")
    return


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=str)
    parser.add_argument("--edit-dir", required=True, type=str)
    parser.add_argument("--pic-extension", default="png", type=str)
    parser.add_argument("--original-caption", default="", type=str)
    parser.add_argument("--edited-caption", default="", type=str)
    parser.add_argument("--start-iter", default=2000, type=int)
    parser.add_argument("--end-iter", default=10000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--fig-title", default="", type=str)
    args = parser.parse_args()
    args.input_dir = Path(args.input_dir)
    args.edit_dir = Path(args.edit_dir)
    # compute_mean_clip_similarity(args)
    iteration_similarity_plot(args)


if __name__=="__main__":
    main()
