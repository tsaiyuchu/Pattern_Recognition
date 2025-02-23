# Pattern_Recognition
Evaluates the mAP and inference time performance of YOLOv3, YOLOv4, RetinaNet, Faster R-CNN, and Mask R-CNN on the Kitti dataset.
# Object Detection on KITTI Dataset

## Dataset Preparation
This project utilizes various object detection models (YOLOv3, YOLOv4, RetinaNet, Faster R-CNN, and Mask R-CNN) on the KITTI dataset.

### Convert KITTI to COCO Format
1. Use `kitti2coco-label-trans.py` to convert KITTI annotations to COCO format:
   ```bash
   python PyTorch-YOLOv3-kitti-master/label_transform/kitti2coco-label-trans.py
   ```
2. Convert COCO format to JSON using `coco2json_modify.py`:
   ```bash
   python coco2json_modify.py
   ```
3. Place the generated `train.json`, `test.json`, and `val.json` files in the appropriate directories.

---

## YOLOv3

### Installation
```bash
# Clone repository
cd ..
pip install -r requirements.txt
```

### Training
```bash
python train.py
```

### Testing
```bash
python test.py
```

### Detection
```bash
python detect.py
```

---

## YOLOv4

### Installation
```bash
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
make
pip install -r requirements.txt
```

### Training
```bash
darknet detector train C:/Proposal/darknet/build/darknet/x64/data/kitty.data \
C:/Proposal/darknet/build/darknet/x64/cfg/yolov3_kitty.cfg \
C:/Proposal/Stanford_Dogs_Dataset/Trainingfiles/yolov4.conv.137 \
-mjpeg_port 8090 -map -gpus 0 -dont_show
```

### mAP Calculation
```bash
darknet detector map C:/Proposal/darknet/build/darknet/x64/data/kitty.data \
C:/Proposal/darknet/build/darknet/x64/cfg/yolov3_kitty.cfg \
C:/Proposal/darknet/build/darknet/x64/backup/yolov4_kitty_best.weights
```

### Testing
Ensure `yolov4_patch_test.py` is placed in the same directory as `darknet.exe`.
```bash
python yolov4_patch_test.py C:/Proposal/darknet/build/darknet/x64/cfg/yolov3_kitty.cfg \
C:/Proposal/darknet/build/darknet/x64/backup/yolov4_kitty_best.weights C:/kitty/yolov4/test.txt
```

---

## Detectron2

### Installation
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python
pip install ninja
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

### Troubleshooting Compilation Issues
```bash
cd path_to_detectron2
python -m pip install -e .
cd ..
python -m pip install -e detectron2
```
If issues persist, ensure all paths are correctly set. Modify `setup.py` to include:
```python
'-allow-unsupported-compiler',  # Add this flag
```

---

## Model Switching
Modify `detectron2_train.py` to change the model:

### RetinaNet
```python
args.config_file = "E:/testdect/detectron2/configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml"
cfg.MODEL.WEIGHTS = "E:/testdect/detectron2/modelweight/model_final_5bd44e.pkl"
```

### Faster R-CNN
```python
args.config_file = "E:/testdect/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
cfg.MODEL.WEIGHTS = "E:/testdect/detectron2/modelweight/model_final_280758.pkl"
```

### Mask R-CNN
```python
args.config_file = "E:/testdect/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
cfg.MODEL.WEIGHTS = "E:/testdect/detectron2/modelweight/model_final_f10217.pkl"
```

---

## Commands

### Training
```bash
python detectron2_train.py --num-gpus 1
```

### Testing
```bash
python detectron2_train.py --eval-only
