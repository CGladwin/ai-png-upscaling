import cv2
import argparse
from cv2 import dnn_superres

# TODO: 
'''
def determine_scaling(initial_image_width: int): -> int
'''

def denoise_image(image, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21): 
    # Apply the fastNlMeansDenoisingColored algorithm
    return cv2.fastNlMeansDenoisingColored(image, None, h, hColor, templateWindowSize, searchWindowSize)

def upscale_image(input_path, output_path, model_name='edsr', scale=4, model_path=None):
        # Read the input image (supports PNG and other formats)
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Could not load image at {}".format(input_path))
    
    image = denoise_image(image)
        
    if scale <1:
        if cv2.imwrite(output_path, image):
            print(f"No upscaling performed. Output image saved to '{output_path}'.")
        else:
            print(f"Failed to save output image to '{output_path}'.")
        return
    
    # Create the super resolution object
    sr = dnn_superres.DnnSuperResImpl_create()

    # Determine model file
    if model_path is None:
        # Assume the model file is named like 'EDSR_x4.pb' (case-sensitive)
        model_file = f"{model_name.upper()}_x{scale}.pb"
    else:
        model_file = model_path

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
    parser = argparse.ArgumentParser(description="Upscale a PNG image using OpenCV's DNN Super Resolution.")
    parser.add_argument("input", help="Path to the input PNG image.")
    parser.add_argument("output", help="Path where the upscaled image will be saved.")
    parser.add_argument("--model", default="edsr", choices=['edsr', 'fsrcnn', 'espcn'],
                        help="Super resolution model to use (default: edsr).")
    parser.add_argument("--scale", type=int, default=4, help="Upscaling factor (default: 4).")
    parser.add_argument("--model_path", default=None,
                        help="Optional path to the model file (e.g., EDSR_x4.pb).")
    args = parser.parse_args()

    upscale_image(args.input, args.output, args.model, args.scale, args.model_path)
