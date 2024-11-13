import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from ultralytics import YOLO

model = YOLO('D:/KHKT24-25/trainmodel/runs/detect/train6/weights/best.pt')

if __name__ == "__main__":
# results = model("Untitled.png", save = True)
    # model.train(data = 'D:/KHKT24-25/trainmodel/data.yaml', epochs = 500, device ='0', batch = 8, workers = 4)
    results = model.track(source = "video.mp4", show = True)