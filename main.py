
import cv2 as cv
import numpy as np
import argparse
import os
import torch
import torch.optim as optim
from utils.yolov1_utils import non_max_suppression, cellboxes_to_boxes
import torchvision.transforms as T
from models.yolov1net_vgg19bn import YoloV1_Vgg19bn
from models.yolov1net_resnet18 import YoloV1_Resnet18
from models.yolov1net_resnet50 import YoloV1_Resnet50
from models.tiny_yolov1net_mobilenetv3_large import Tiny_YoloV1_MobileNetV3_Large
from models.tiny_yolov1net_mobilenetv3_small import Tiny_YoloV1_MobileNetV3_Small
from models.tiny_yolov1net_squeezenet import Tiny_YoloV1_SqueezeNet
from utils.custom_transform import draw_bounding_box

MODEL_DICT = {
    "vgg19": YoloV1_Vgg19bn,
    "resnet18": YoloV1_Resnet18,
    "resnet50": YoloV1_Resnet50,
    "mobilenetv3-large": Tiny_YoloV1_MobileNetV3_Large,
    "mobilenetv3-small": Tiny_YoloV1_MobileNetV3_Small,
    "squeezenet": Tiny_YoloV1_SqueezeNet
}

def main():
    parser = argparse.ArgumentParser(description="Inference YOLOv1 on a folder containing images.")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_DICT.keys(),
                        help="Choose a YOLOv1 model to use for inference(vgg19, resnet18, resnet50, mobilenetv3-large, mobilenetv3-small, squeezenet)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Confidence threshold for confident predictions (default: 0.4).")
    args = parser.parse_args()
   
    IMAGE_FOLDER = "images" 
    RESULTS_FOLDER = "results"
    os.makedirs(RESULTS_FOLDER, exist_ok=True)  

    transform = T.Compose([T.ToTensor()])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device using: ", device)


    model_class = MODEL_DICT[args.model]
    model = model_class(S=7, B=2, C=20).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0005)
    if args.model == "vgg19":
        path_cpt_file = f"pretrained/vgg19bn_adj_lr_yolov1.cpt"
    elif args.model == "resnet18":
        path_cpt_file = f"pretrained/resnet18_adj_lr_yolov1.cpt"
    elif args.model == "resnet50":
        path_cpt_file = f"pretrained/resnet50_adj_lr_yolov1.cpt"
    elif args.model == "mobilenetv3-large":
        path_cpt_file = f"pretrained/mobilenetv3-large_tiny_adj_lr_yolov1.cpt"
    elif args.model == "mobilenetv3-small":
        path_cpt_file = f"pretrained/mobilenetv3-small_tiny_adj_lr_yolov1.cpt"
    elif args.model == "squeezenet":
        path_cpt_file = f"pretrained/squeezenet_tiny_adj_lr_yolov1.cpt"
    
    if not os.path.exists(path_cpt_file):
        print(f"Error: Checkpoint file {path_cpt_file} not found.")
        return
    
    checkpoint = torch.load(path_cpt_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    print("Pretrained YOLOv1 model loaded.")

    for img_name in os.listdir(IMAGE_FOLDER):
        if img_name.lower().endswith((".png", ".jpg", ".jpeg")): 
            img_path = os.path.join(IMAGE_FOLDER, img_name)
            print(f"Processing {img_name}...")

            frame = cv.imread(img_path)
            frame = cv.resize(frame, (448, 448))
            orig_frame = frame.copy()

            frame = transform(frame)    
            frame = frame.unsqueeze(0) 
            preds = model(frame.to(device))  

            get_bboxes = cellboxes_to_boxes(preds)
            bboxes = non_max_suppression(get_bboxes[0], iou_threshold=0.5, threshold=args.conf, boxformat="midpoints")

            result_frame = draw_bounding_box(orig_frame, bboxes, test=True)

            result_path = os.path.join(RESULTS_FOLDER, img_name)
            cv.imwrite(result_path, result_frame)
            print(f"Saved result to {result_path}")


    cv.destroyAllWindows()
    print("Inference completed.")

if __name__ == "__main__":
    main()