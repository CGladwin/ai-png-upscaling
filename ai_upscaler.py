import cv2
import argparse
from cv2 import dnn_superres

def denoise_image(image):
    if DENOISE_STRENGTH < 1:
        return image
    image = cv2.bilateralFilter(image,9,75,75)
    if DENOISE_STRENGTH >= 2:
        image = cv2.medianBlur(image,5)
    if DENOISE_STRENGTH >= 3:
        image = cv2.fastNlMeansDenoisingColored(image,None,4,4,7,21)
    return image


def upscale_image(input_path, output_path):
        # Read the input image (supports PNG and other formats)
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Could not load image at {}".format(input_path))
    
    image = denoise_image(image)
    # cv2.imwrite(output_path, image)
    # return
        
    if SCALE <1:
        if cv2.imwrite(output_path, image):
            print(f"No upscaling performed. Output image saved to '{output_path}'.")
        else:
            print(f"Failed to save output image to '{output_path}'.")
        return
    
    # Create the super resolution object
    sr = dnn_superres.DnnSuperResImpl_create()

    # Load the pre-trained model
    try:
        sr.readModel(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model file '{MODEL_PATH}': {e}")
        return

    # Set the model and scale
    sr.setModel(MODEL_NAME, SCALE)

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
    MODEL_NAME= 'edsr'
    SCALE = args.scale  #in case I want to add models to expand this feature
    MODEL_PATH  = args.model_path if args.model_path is not None else f"{MODEL_NAME.upper()}_x{SCALE}.pb"

    upscale_image(args.input, args.output)
