import cv2
import argparse
from cv2 import dnn_superres
import numpy as np

# TODO: 
'''
def determine_scaling(initial_image_width: int): -> int
'''

def denoise_image(image):
    if DENOISE_STRENGTH < 1:
        return image
    image = cv2.bilateralFilter(image,9,75,75)
    if DENOISE_STRENGTH >= 2:
        image = cv2.medianBlur(image,5)
    if DENOISE_STRENGTH >= 3:
        image = cv2.fastNlMeansDenoisingColored(image,None,4,4,7,21)
    return image


def upscale_image(input_path, output_path, model_name='edsr', scale=4):
        # Read the input image (supports PNG and other formats)
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Could not load image at {}".format(input_path))
    
    image = denoise_image(image)
    # cv2.imwrite(output_path, image)
    # return
        
    if scale <1:
        if cv2.imwrite(output_path, image):
            print(f"No upscaling performed. Output image saved to '{output_path}'.")
        else:
            print(f"Failed to save output image to '{output_path}'.")
        return
    
    # Create the super resolution object
    sr = dnn_superres.DnnSuperResImpl_create()

    # Determine model file
    if MODEL_PATH is None:
        # Assume the model file is named like 'EDSR_x4.pb' (case-sensitive)
        model_file = f"{model_name.upper()}_x{scale}.pb"
    else:
        model_file = MODEL_PATH

    # Load the pre-trained model
    try:
        sr.readModel(model_file)
    except Exception as e:
        print(f"Error loading model file '{model_file}': {e}")
        return

    # Set the model and scale
    sr.setModel(model_name, scale)

    # Perform upscaling
    upscaled = sr.upsample(image)

    # Save the upscaled image
    if cv2.imwrite(output_path, upscaled):
        print(f"Upscaled image saved to '{output_path}'.")
    else:
        print(f"Failed to save upscaled image to '{output_path}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Denoise and AI Upscale a PNG using OpenCV.")
    parser.add_argument("input", help="Path to the input PNG image.")
    parser.add_argument("output", help="Path where the upscaled image will be saved.")
    parser.add_argument("--denoise", type=int, default=0, help="level of denoising to apply before upscaling (from 0 to 3)")
    parser.add_argument("--scale", type=int, default=4, help="Upscaling factor (default: 4).")
    parser.add_argument("--model_path", default=None,
                        help="Optional path to the model file (e.g., EDSR_x4.pb).")
    args = parser.parse_args()
    DENOISE_STRENGTH = args.denoise
    MODEL_PATH  = args.model_path
    upscale_image(args.input, args.output, args.scale)
