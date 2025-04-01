
import argparse
import torch
from PIL import Image
from realesrgan import RealESRGAN

def upscale_image(input_path, output_path, scale=4):
    # Determine the device to use (GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize the Real-ESRGAN model for the given scale
    model = RealESRGAN(device, scale=scale)
    
    # Load the pretrained weights (ensure you have the proper .pth file in your working directory)
    # For example, for scale=4, you would use 'RealESRGAN_x4.pth'
    weights_path = f'RealESRGAN_x{scale}.pth'
    try:
        model.load_weights(weights_path)
    except Exception as e:
        print(f"Error loading weights from {weights_path}: {e}")
        return

    # Open the image and ensure it's in RGB mode (the model expects RGB input)
    try:
        image = Image.open(input_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Perform super-resolution
    try:
        sr_image = model.predict(image)
    except Exception as e:
        print(f"Error during upscaling: {e}")
        return

    # Save the upscaled image
    try:
        sr_image.save(output_path)
        print(f"Upscaled image saved to {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Upscale a PNG image using Real-ESRGAN AI model.")
    parser.add_argument("input", help="Path to the input PNG image.")
    parser.add_argument("output", help="Path where the upscaled image will be saved.")
    parser.add_argument("--scale", type=int, default=4, help="Upscaling factor (default: 4).")
    args = parser.parse_args()

    upscale_image(args.input, args.output, args.scale)
