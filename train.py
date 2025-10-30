from ultralytics import YOLO
import torch

def train_model():
    # Check for GPU
    if not torch.cuda.is_available():
        print("CUDA GPU not found. Training will be slow.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Load a pre-trained YOLOv8 segmentation model
    model = YOLO("yolov8n.pt")  # instead of yolov8n-seg.pt


    # Train the model
    print("Starting model training...")
    results = model.train(
        data='C:\\Users\\aswat\\OneDrive\\Desktop\\Project\\RepairMate\\lap.v2i.yolov11\\data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        name='laptop_parts_segmentation'
    )
    print(f"Training complete. Results saved to: {results.save_dir}")

if __name__ == '__main__':
    train_model()