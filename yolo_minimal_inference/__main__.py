import sys
from imageio import imread
from yolo_minimal_inference.yolo import YOLO
def main():
    if len(sys.argv) != 3:
        print("Usage: python -m yolo_minimal_inference <model_path> <image_path>")
        sys.exit(1)

    model_path, image_path = sys.argv[1], sys.argv[2]

    # Load the model
    yolo = YOLO(model_path,is_brg=False)

    # Read the image
    image = imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        sys.exit(1)

    # Run inference
    outputs = yolo(image)
    print("Model Outputs:", outputs)

if __name__ == "__main__":
    main()
