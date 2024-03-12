import os
from argparse import ArgumentParser
from PIL import Image
import numpy as np

def resize_images(source_dir, target_dir, target_size, image_type):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for file_name in os.listdir(source_dir):
        if file_name.endswith('.' + image_type):
            file_path = os.path.join(source_dir, file_name)
            with Image.open(file_path) as img:
                # Resize the image to the target size without preserving aspect ratio
                resized_img = img.resize(target_size)
                
                # Save the resized image
                save_path = os.path.join(target_dir, file_name)
                resized_img.save(save_path, image_type.upper())

def get_file_names(directory, extension):
    """Fetch all file names with a specific extension from the given directory."""
    return [file for file in os.listdir(directory) if file.endswith(extension) and os.path.isfile(os.path.join(directory, file))]


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--width", default=512, type=int)
    parser.add_argument("--height", default=512, type=int)
    parser.add_argument("--pic_extension", default="png", type=str)
    parser.add_argument("--cam_extension", default="txt", type=str)
    
    input_width, input_height = 1600, 1200

    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    pic_extension = args.pic_extension
    cam_extension = args.cam_extension
    height = args.height
    width = args.width
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    resize_images(input_dir, output_dir, (width, height), pic_extension)
    
    input_files = get_file_names(input_dir, f".{cam_extension}")
    for file in input_files:
        txt = np.loadtxt(fname=f"{input_dir}/{file}")
        width_ratio = width/input_width
        heigth_ratio = height/input_height


        if file.endswith("k.txt"):
            txt = np.multiply(txt, [[width_ratio, width_ratio, width_ratio],[1, heigth_ratio, heigth_ratio],[1, 1, 1]])
        np.savetxt(f"{output_dir}/{file}", txt, fmt="%.20f")

if __name__:
    main()
