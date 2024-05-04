import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import os
import cv2



def convert_image_to(input_folder, output_folder, operation_type="IMREAD_GRAYSCALE", canny=False, threshold1=100, threshold2=200):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Determine the correct flag to use based on operation_type
    if hasattr(cv2, operation_type):
        read_flag = getattr(cv2, operation_type)
    else:
        raise ValueError(f"Invalid operation type: {operation_type}")

    for filename in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, filename)
        
        if os.path.isfile(input_file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            # Read the image with the specified operation type
            image = cv2.imread(input_file_path, read_flag)
            if image is None:
                print(f"Failed to read {filename} as an image.")
                continue

            # Convert to grayscale if necessary (Canny requires grayscale images)
            if canny or operation_type != "IMREAD_GRAYSCALE":
                if image.ndim == 3:  # Color image
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection if requested
            if canny:
                image = cv2.Canny(image, threshold1, threshold2)

            output_file_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_file_path, image)
            print(f"Converted {filename} and saved to {output_folder}")


def show_image(image):
  """
  This function shows an image.

  Args:
    image: An image tensor.
  """

  # Convert the image tensor to a PIL Image.
  img = np.array(image)

  # Show the image.
  plt.imshow(image)
  plt.show()

def show_tensor_image(tensor):
    # Check if the tensor has 3 or 1 channel
    if tensor.shape[0] == 3:
        # Convert from tensor shape (C, H, W) to numpy shape (H, W, C)
        numpy_image = tensor.permute(1, 2, 0).numpy()
    elif tensor.shape[0] == 1:
        # Convert from tensor shape (1, H, W) to numpy shape (H, W)
        numpy_image = tensor.squeeze().numpy()
    else:
        raise ValueError("Tensor has invalid number of channels. Expected 1 or 3, got {}".format(tensor.shape[0]))

    # Denormalize if necessary (assuming normalization was mean=0.5, std=0.5)
    # Adjust these values if you used different values for normalization
    # numpy_image = numpy_image * 0.5 + 0.5

    # # Clip the values to be between 0 and 1
    # numpy_image = np.clip(numpy_image, 0, 1)

    # Display the image
    plt.imshow(numpy_image)
    plt.axis('off')  # Turn off axis numbers
    plt.show()
