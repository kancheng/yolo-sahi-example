# !pip install -U torch sahi yolov5 imantics fiftyone

import os
p = os.getcwd()

# arrange an instance segmentation model for test
from sahi.utils.yolov5 import (
    download_yolov5s6_model,
)

# import required functions, classes
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image

# download YOLOV5S6 model to 'models/yolov5s6.pt'
yolov5_model_path =  p + '/models/yolov5s6.pt'
download_yolov5s6_model(destination_path=yolov5_model_path)

# download test images into demo_data folder
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg', 'demo_data_v5/small-vehicles1.jpeg')
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png', 'demo_data_v5/terrain2.png')

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov5',
    model_path=yolov5_model_path,
    confidence_threshold=0.3,
    device="cpu", # or 'cuda:0'
)

result = get_prediction("demo_data_v5/small-vehicles1.jpeg", detection_model)
result = get_prediction(read_image("demo_data_v5/small-vehicles1.jpeg"), detection_model)
result.export_visuals(export_dir="demo_data_v5/")
Image("demo_data_v5/prediction_visual.png")

result = get_sliced_prediction(
    "demo_data_v5/small-vehicles1.jpeg",
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)


result.export_visuals(export_dir="demo_data_v5/")
Image("demo_data_v5/prediction_visual.png")

object_prediction_list = result.object_prediction_list
object_prediction_list[0]
result.to_coco_annotations()[:3]

result.to_coco_predictions(image_id=1)[:3]

result.to_imantics_annotations()[:3]

result.to_fiftyone_detections()[:3]

model_type = "yolov5"
model_path = yolov5_model_path
model_device = "cpu" # or 'cuda:0'
model_confidence_threshold = 0.4

slice_height = 256
slice_width = 256
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2

source_image_dir = "demo_data_v5/"


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

exit() 