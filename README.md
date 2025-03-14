# YOLOV1  -  You Only Look Once 

## Description
This is a project to run inference of pretrained YOLOV1 models on pytorch.

![Image](https://github.com/user-attachments/assets/a5313a9b-37c8-46f0-b3fb-13f971233ac3)
## 🚀 Quick Start

### Requirements
I recommend you to use python >= 3.9 to run project.

### **1️⃣ Clone the Project**

Clone with HTTPS
```bash
  git clone https://github.com/kendyle2702/yolov1-inference.git
  cd pis
```
Clone with SSH
```bash
  git clone git@github.com:kendyle2702/yolov1-inference.git
  cd pis
```

### **2️⃣ Install Library**
```bash
  pip install -r requirements.txt
```

### **3️⃣ Download Pretrained Model**

| Model | Link |
|-------|-------|
| Vgg19 | [Link](https://drive.google.com/file/d/1-5vqoN2QxRqvFQ_KBZxD2q3hi5dBwcmq/view?usp=sharing) |
| Resnet18 |[Link](https://drive.google.com/file/d/1VsDFNMDYBWSy9qFGooMVNo5SFVyYToT0/view?usp=sharing) |
| Resnet50 | [Link](https://drive.google.com/file/d/1-31xnUeXpkb2AHLr9GDw0wlgn9MmUM13/view?usp=sharing) |
| Tiny Mobilenetv3 Small | [Link](https://drive.google.com/file/d/1-i-V_hXNPH75I-PpErn3bZRLdtDNVlFO/view?usp=sharing)|
| Tiny Mobilenetv3 Large | [Link](https://drive.google.com/file/d/1-lYeKLf3pE2wmUb_TaNIRnrzdn8TubBZ/view?usp=sharing)|
|Tiny Squeezenet | [Link](https://drive.google.com/file/d/1-ZO32j6K7L41qpnwXTeRS0LvJY_bV9lL/view?usp=sharing)|

Move pretrained model to ```pretrained``` folder. 


### **4️⃣ Run Inference**
Make sure you have put the images you need to inference into the ```images``` folder.
```bash
  python main.py --model {vgg19, resnet18, resnet50, mobilenetv3-large, mobilenetv3-small, squeezenet} 
                 --conf {default 0.4}
```
The image inference results will be in the ```results``` folder.
