from pathlib import Path

from ultralytics import YOLO

def main():
    DATA_DIR = Path("Data")

    model = YOLO("yolo11.pt")
    model.to('cuda')

    model.train(
        data=DATA_DIR / "data.yaml",
        epochs=20,
        imgsz=640,
        batch=-1,
        workers=1
    )
    model.save('trained.pt')


if __name__ == '__main__':
    main()
