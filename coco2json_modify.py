#pip install pillow
import os
import json
from PIL import Image

# 設定資料集路徑
IMAGE_DIR = "E:/detectron2/datasets/kitti/images/train"
LABEL_DIR = "E:/detectron2/datasets/kitti/labels/train"
OUTPUT_JSON = "E:/detectron2/datasets/kitti/json/train/train_corrected.json"

# 類別名稱
CLASS_NAMES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]

def convert_bbox_to_coco_format(bbox, img_width, img_height):
    x_center, y_center, width, height = bbox
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x = x_center - width / 2
    y = y_center - height / 2
    return [x, y, width, height]

def convert_bbox_to_segmentation(bbox, img_width, img_height):
    x_center, y_center, width, height = bbox
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x = x_center - width / 2
    y = y_center - height / 2
    return [[x, y, x + width, y, x + width, y + height, x, y + height]]

# 構建 COCO 格式的 JSON 文件
coco_output = {
    "images": [],
    "annotations": [],
    "categories": [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES)]
}

annotation_id = 1
for image_id, image_filename in enumerate(os.listdir(IMAGE_DIR)):
    if not image_filename.endswith(".png"):
        continue
    
    image_path = os.path.join(IMAGE_DIR, image_filename)
    label_path = os.path.join(LABEL_DIR, os.path.splitext(image_filename)[0] + ".txt")
    
    with Image.open(image_path) as img:
        width, height = img.size
    
    with open(label_path, "r") as label_file:
        img_info = {
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        }
        coco_output["images"].append(img_info)
        
        for line in label_file:
            parts = line.strip().split()
            category_id = int(parts[0])
            bbox = list(map(float, parts[1:]))
            coco_bbox = convert_bbox_to_coco_format(bbox, width, height)
            segmentation = convert_bbox_to_segmentation(bbox, width, height)
            area = coco_bbox[2] * coco_bbox[3]
            
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": coco_bbox,
                "segmentation": segmentation,
                "area": area,
                "iscrowd": 0
            }
            coco_output["annotations"].append(annotation)
            annotation_id += 1

with open(OUTPUT_JSON, "w") as json_file:
    json.dump(coco_output, json_file)
