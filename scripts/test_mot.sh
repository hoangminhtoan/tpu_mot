ROOT_DIR="/home/mendel/Workspace/VOT/fastMOT"

python3 $ROOT_DIR/src/mot.py \
--detector_model $ROOT_DIR/weights/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
--extractor_model $ROOT_DIR/weights/osnet_x0_25_msmt17_quant_tpu.tflite \
--labels $ROOT_DIR/weights/coco_labels.txt \
--input_data  /dev/video1