from utils import convert_image_to
type = "canny"
def image_converter():
    # Example usage:
    input_folder = 'data/main/'
    output_folder = ["data/train/images", "data/train/labels", ]

    convert_image_to(input_folder, output_folder[0], canny=False)
    convert_image_to(input_folder, output_folder[1], canny=True)


if __name__ == "__main__":
    image_converter()