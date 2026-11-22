# Object detection with auto-labeled training data and RF-DETR

Auto-label training images with the open-vocabulary [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) detector, then fine-tune [RF-DETR Nano](https://github.com/roboflow/rf-detr) on the pseudo-labels for 15 epochs on one A10 GPU. The [People & Luggage Detection](https://huggingface.co/datasets/abhinand4as/people-luggage-detection) dataset has two classes, `luggage` and `person`.

The recipe ignores the dataset's human train labels. Human validation labels select the auto-labeler's confidence threshold and RF-DETR's best checkpoint. The threshold is the candidate with the best validation F1 at IoU 0.5. F1 penalizes the extra boxes a low threshold adds. COCO mAP would always favor the lowest threshold. Human test labels are used only to evaluate both models.

## Before you run

- RF-DETR downloads pretrained weights from `repo.roboflow.com`. The workspace needs outbound access to that host. The OWLv2 weights and dataset are downloaded from Hugging Face.
- Review the dataset's CC BY 4.0 license and the models' Apache-2.0 licenses. Both models and the dataset are public and ungated. No Hugging Face token is needed.
- The output path defaults to `/Volumes/main/default/air_examples/object-detection-autolabel-rf-detr`. Set `parameters.output_root` in `workload.yaml` to another existing writable volume directory if needed.

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The AIR MLflow run contains the selected threshold, test mean average precision (mAP) for each model, pseudo-label count, and the output artifacts. `<output_root>/<MLFLOW_RUN_ID>/` contains:

- `pseudo_labels_train.json`: pseudo-labels in COCO format, with a confidence `score` on each box.
- `rfdetr_best.pth`: RF-DETR Nano weights selected on validation.
- `metrics.json`: the threshold sweep and all reported metrics.

The MLflow run also has RF-DETR's `metrics.csv`, with per-epoch training loss and validation mAP.

A run on one A10 took 14 minutes in the container. Validation F1 at IoU 0.5 peaked at threshold 0.2 (precision 0.50, recall 0.55, F1 0.52); at 0.05, F1 was 0.23. At 0.2 the auto-labeler produced 19,557 pseudo-label boxes for 1,456 training images, about 13 per image, against about 10 per image in the human validation labels. The table reports COCO box mean average precision (mAP) on the full test split, averaged across classes and, for mAP@[.50:.95], IoU thresholds from 0.50 to 0.95.

| Model | Test mAP@[.50:.95] | Test mAP@.50 |
| --- | --- | --- |
| OWLv2 auto-labeler at threshold 0.2 | 0.153 | 0.357 |
| RF-DETR Nano on pseudo-labels | 0.235 | 0.507 |

The auto-labeler's mAP is lower at 0.2 than it would be at a lower threshold, because mAP does not penalize extra low-scoring boxes. The student trained on the 0.2 labels outperforms the labeler on both metrics.
