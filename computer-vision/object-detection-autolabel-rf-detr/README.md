# Object detection with auto-labeled training data and RF-DETR

Auto-label training images with the open-vocabulary [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) detector, then fine-tune [RF-DETR Nano](https://github.com/roboflow/rf-detr) on the pseudo-labels for 15 epochs on one A10 GPU. The [People & Luggage Detection](https://huggingface.co/datasets/abhinand4as/people-luggage-detection) dataset has two classes (`luggage` and `person`); this recipe ignores its human train labels and uses the human validation labels to select the auto-labeler's confidence threshold and RF-DETR's best checkpoint. The human test labels are used only to evaluate both models.

## Before you run

- RF-DETR downloads pretrained weights from `repo.roboflow.com`; the workspace needs outbound access to that host. The OWLv2 weights and dataset are downloaded from Hugging Face.
- Review the dataset's CC BY 4.0 license and the models' Apache-2.0 licenses. Both models and the dataset are public and ungated; no Hugging Face token is needed.
- The output path defaults to `/Volumes/main/default/air_examples/object-detection-autolabel-rf-detr`. Set `parameters.output_root` in `workload.yaml` to another existing writable volume directory if needed.

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The AIR MLflow run contains the selected threshold, test mean average precision (mAP) for each model, pseudo-label count, and the output artifacts. `<output_root>/<MLFLOW_RUN_ID>/` contains:

- `pseudo_labels_train.json`: pseudo-labels in COCO format, with a confidence `score` on each box.
- `rfdetr_pseudo_label_best.pth`: RF-DETR Nano weights selected on validation.
- `metrics.json`: the threshold sweep and all reported metrics.
