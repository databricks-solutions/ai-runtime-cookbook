# Recognize actions in video clips

Train a model to recognize 10 actions, including archery, basketball, and applying makeup, from a public [UCF101 subset](https://huggingface.co/datasets/sayakpaul/ucf101-subset) containing 405 videos. On one A10 GPU, the recipe trains a small recurrent network (GRU) for 20 epochs using frame features from a frozen, ImageNet-trained [ResNet-18](https://huggingface.co/timm/resnet18.a1_in1k).

PyAV reads 16 evenly spaced frames from each clip. ResNet-18 describes each frame, and the GRU learns from those descriptions in sequence. Frame features are computed once and kept in memory while the classifier trains.

Related clips stay together: the recipe assigns whole video groups to training, validation, or test data, replacing the archive's original clip-level splits. It chooses the best checkpoint using validation accuracy, then evaluates that checkpoint on the test set. These are custom splits of a small subset; results do not represent the full UCF101 benchmark.

## Before you run

Check `parameters.output_root` in [workload.yaml](workload.yaml). You need write access to the existing Unity Catalog volume; change the path if you cannot use the shared default.

The video archive and ResNet-18 weights download automatically without a Hugging Face token. See the [subset dataset card](https://huggingface.co/datasets/sayakpaul/ucf101-subset) and [UCF101 dataset page](https://www.crcv.ucf.edu/research/data-sets/ucf101/) for data provenance and usage terms.

## Run

From this recipe directory:

```bash
databricks air run -f workload.yaml
```

Change `epochs` or `hidden_size` under `parameters` to adjust classifier training. `frames_per_video` controls how much of each clip the model sees; more frames require more decoding work and GPU memory.

## Results

Each run writes its files to `<output_root>/<MLFLOW_RUN_ID>/` and logs them to the AIR MLflow run.

| File | What it contains |
| --- | --- |
| `classifier.pt` | The best GRU classifier weights, class names, and settings needed to rebuild it. ResNet-18 weights are downloaded separately when reusing the model. |
| `test_predictions.csv` | Each test video's action, predicted action, and confidence score. |
| `confusion_matrix.csv` | Counts of predictions, with actual actions as rows and predicted actions as columns. |
| `split_manifest.csv` | Every video's class, original group, and assigned split. |
| `metrics.json` | Test accuracy, macro F1, best validation epoch, and processing times. |

MLflow also tracks training loss and validation accuracy for each epoch.
