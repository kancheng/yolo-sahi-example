## 0. Preperation
print("0. Preperation")
import os
p = os.getcwd()
# arrange an instance segmentation model for test
from sahi.utils.yolov8 import (
    download_yolov8s_model, download_yolov8s_seg_model
)

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image
# download YOLOV5S6 model to 'models/yolov5s6.pt'
yolov8_model_path = p +"/models/yolov8s.pt"
download_yolov8s_model(yolov8_model_path)

# download test images into demo_data folder
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg', 'demo_data_v8/small-vehicles1.jpeg')
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png', 'demo_data_v8/terrain2.png')

## 1. Standard Inference with a YOLOv8 Model
print("1. Standard Inference with a YOLOv8 Model")
# - Instantiate a detection model by defining model weight path and other parameters:
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.3,
    device="cpu", # or 'cuda:0'
)

# - Perform prediction by feeding the get_prediction function with an image path and a DetectionModel instance:

result = get_prediction("demo_data_v8/small-vehicles1.jpeg", detection_model)

# - Or perform prediction by feeding the get_prediction function with a numpy image and a DetectionModel instance:

result = get_prediction(read_image("demo_data_v8/small-vehicles1.jpeg"), detection_model)

# - Visualize predicted bounding boxes and masks over the original image:

result.export_visuals(export_dir="demo_data_v8/")
Image("demo_data_v8/prediction_visual.png")

## 2. Sliced Inference with a YOLOv8 Model
print("2. Sliced Inference with a YOLOv8 Model")
# - To perform sliced prediction we need to specify slice parameters. In this example we will perform prediction over slices of 256x256 with an overlap ratio of 0.2:

result = get_sliced_prediction(
    "demo_data_v8/small-vehicles1.jpeg",
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)

# - Visualize predicted bounding boxes and masks over the original image:

result.export_visuals(export_dir="demo_data_v8/")
Image("demo_data_v8/prediction_visual.png")

## 3. Prediction Result
print("3. Prediction Result")
# - Predictions are returned as [sahi.prediction.PredictionResult](sahi/prediction.py), you can access the object prediction list as:

object_prediction_list = result.object_prediction_list

object_prediction_list[0]

# - ObjectPrediction's can be converted to [COCO annotation](https://cocodataset.org/#format-data) format:

result.to_coco_annotations()[:3]

# - ObjectPrediction's can be converted to [COCO prediction](https://github.com/i008/COCO-dataset-explorer) format:

result.to_coco_predictions(image_id=1)[:3]

# - ObjectPrediction's can be converted to [imantics](https://github.com/jsbroks/imantics) annotation format:

result.to_imantics_annotations()[:3]

# - ObjectPrediction's can be converted to [fiftyone](https://github.com/voxel51/fiftyone) detection format:

result.to_fiftyone_detections()[:3]

## 4. Batch Prediction
print("4. Batch Prediction")
# - Set model and directory parameters:

model_type = "yolov8"
model_path = yolov8_model_path
model_device = "cpu" # or 'cuda:0'
model_confidence_threshold = 0.4

slice_height = 256
slice_width = 256
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2

source_image_dir = "demo_data_v8/"

# - Perform sliced inference on given folder:

predict(
    model_type=model_type,
    model_path=model_path,
    model_device=model_device,
    model_confidence_threshold=model_confidence_threshold,
    source=source_image_dir,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
)

## 5. Segmentation
print("5. Segmentation")
# Run above examples for segmentation model.

# download YOLOV8S model to 'models/yolov8s.pt'
yolov8_seg_model_path =  p +"/models/yolov8s-seg.pt"
# yolov8_seg_model_path = "models/yolov8n-seg.pt"
download_yolov8s_seg_model(yolov8_seg_model_path)

detection_model_seg = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_seg_model_path,
    confidence_threshold=0.3,
    device="cpu", # or 'cuda:0'
)

im = read_image("demo_data_v8/small-vehicles1.jpeg")
h = im.shape[0]
w = im.shape[1]

result = get_prediction("demo_data_v8/small-vehicles1.jpeg", detection_model_seg, full_shape=(h, w))

result.export_visuals(export_dir="demo_data_v8/")

Image("demo_data_v8/prediction_visual.png")

result = get_sliced_prediction(
    "demo_data_v8/small-vehicles1.jpeg",
    detection_model_seg,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)

result.export_visuals(export_dir="demo_data_v8/")

Image("demo_data_v8/prediction_visual.png")

object_prediction_list = result.object_prediction_list
object_prediction_list[0]

object_prediction_list[0].mask.segmentation

# Sliced predictions are much better

predict(
    model_type=model_type,
    model_path=model_path,
    model_device=model_device,
    model_confidence_threshold=model_confidence_threshold,
    source=source_image_dir,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
)


