# Useful Tools

Apart from training/testing scripts, We provide lots of useful tools under the
`tools/` directory.

## Analysis Tools

### Plot training logs

`tools/analyze_logs.py` plots loss/mIoU curves given a training log file. `pip install seaborn` first to install the dependency.

```shell
python tools/analysis_tools/analyze_logs.py xxx.json [--keys ${KEYS}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

Examples:

- Plot the mIoU, mAcc, aAcc metrics.

  ```shell
  python tools/analysis_tools/analyze_logs.py log.json --keys mIoU mAcc aAcc --legend mIoU mAcc aAcc
  ```

- Plot loss metric.

  ```shell
  python tools/analysis_tools/analyze_logs.py log.json --keys loss --legend loss
  ```

### Confusion Matrix (experimental)

In order to generate and plot a `nxn` confusion matrix where `n` is the number of classes, you can follow the steps:

#### 1.Generate a prediction result in pkl format using `test.py`

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${PATH_TO_RESULT_FILE}]
```

Example:

```shell
python tools/test.py \
configs/fcn/fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py \
checkpoint/fcn_r50-d8_512x1024_40k_cityscapes_20200604_192608-efe53f0d.pth \
--out result/pred_result.pkl
```

#### 2. Use `confusion_matrix.py` to generate and plot a confusion matrix

```shell
python tools/confusion_matrix.py ${CONFIG_FILE} ${PATH_TO_RESULT_FILE} ${SAVE_DIR} --show
```

Description of arguments:

- `config`: Path to the test config file.
- `prediction_path`: Path to the prediction .pkl result.
- `save_dir`: Directory where confusion matrix will be saved.
- `--show`: Enable result visualize.
- `--color-theme`: Theme of the matrix color map.
- `--cfg_options`: Custom options to replace the config file.

Example:

```shell
python tools/confusion_matrix.py \
configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py \
result/pred_result.pkl \
result/confusion_matrix \
--show
```

### Get the FLOPs and params (experimental)

We provide a script adapted from [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) to compute the FLOPs and params of a given model.

```shell
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

You will get the result like this.

```none
==============================
Input shape: (3, 2048, 1024)
Flops: 1429.68 GMac
Params: 48.98 M
==============================
```

:::{note}
This tool is still experimental and we do not guarantee that the number is correct. You may well use the result for simple comparisons, but double check it before you adopt it in technical reports or papers.
:::

(1) FLOPs are related to the input shape while parameters are not. The default input shape is (1, 3, 1280, 800).
(2) Some operators are not counted into FLOPs like GN and custom operators.

## Miscellaneous

### Publish a model

Before you upload a model to AWS, you may want to
(1) convert model weights to CPU tensors, (2) delete the optimizer states and
(3) compute the hash of the checkpoint file and append the hash id to the filename.

```shell
python tools/misc/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/publish_model.py work_dirs/pspnet/latest.pth psp_r50_512x1024_40k_cityscapes.pth
```

The final output filename will be `psp_r50_512x1024_40k_cityscapes-{hash id}.pth`.

### Print the entire config

`tools/misc/print_config.py` prints the whole config verbatim, expanding all its
imports.

```shell
python tools/misc/print_config.py \
  ${CONFIG} \
  --graph \
  --cfg-options ${OPTIONS [OPTIONS...]} \
```

Description of arguments:

- `config` : The path of a pytorch model config file.
- `--graph` : Determines whether to print the models graph.
- `--cfg-options`: Custom options to replace the config file.

## Model conversion

`tools/model_converters/` provide several scripts to convert pretrain models released by other repos to MMSegmentation style.

### ViT Swin MiT Transformer Models

- ViT

  `tools/model_converters/vit2mmseg.py` convert keys in timm pretrained vit models to MMSegmentation style.

  ```shell
  python tools/model_converters/vit2mmseg.py ${SRC} ${DST}
  ```

- Swin

  `tools/model_converters/swin2mmseg.py` convert keys in official pretrained swin models to MMSegmentation style.

  ```shell
  python tools/model_converters/swin2mmseg.py ${SRC} ${DST}
  ```

- SegFormer

  `tools/model_converters/mit2mmseg.py` convert keys in official pretrained mit models to MMSegmentation style.

  ```shell
  python tools/model_converters/mit2mmseg.py ${SRC} ${DST}
  ```

## Model Serving

In order to serve an `MMSegmentation` model with [`TorchServe`](https://pytorch.org/serve/), you can follow the steps:

### 1. Convert model from MMSegmentation to TorchServe

```shell
python tools/torchserve/mmseg2torchserve.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
--output-folder ${MODEL_STORE} \
--model-name ${MODEL_NAME}
```

:::{note}
${MODEL_STORE} needs to be an absolute path to a folder.
:::

### 2. Build `mmseg-serve` docker image

```shell
docker build -t mmseg-serve:latest docker/serve/
```

### 3. Run `mmseg-serve`

Check the official docs for [running TorchServe with docker](https://github.com/pytorch/serve/blob/master/docker/README.md#running-torchserve-in-a-production-docker-environment).

In order to run in GPU, you need to install [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). You can omit the `--gpus` argument in order to run in CPU.

Example:

```shell
docker run --rm \
--cpus 8 \
--gpus device=0 \
-p8080:8080 -p8081:8081 -p8082:8082 \
--mount type=bind,source=$MODEL_STORE,target=/home/model-server/model-store \
mmseg-serve:latest
```

[Read the docs](https://github.com/pytorch/serve/blob/072f5d088cce9bb64b2a18af065886c9b01b317b/docs/rest_api.md) about the Inference (8080), Management (8081) and Metrics (8082) APIs

### 4. Test deployment

```shell
curl -O https://raw.githubusercontent.com/open-mmlab/mmsegmentation/master/resources/3dogs.jpg
curl http://127.0.0.1:8080/predictions/${MODEL_NAME} -T 3dogs.jpg -o 3dogs_mask.png
```

The response will be a ".png" mask.

You can visualize the output as follows:

```python
import matplotlib.pyplot as plt
import mmcv
plt.imshow(mmcv.imread("3dogs_mask.png", "grayscale"))
plt.show()
```

You should see something similar to:

![3dogs_mask](../../resources/3dogs_mask.png)

And you can use `test_torchserve.py` to compare result of torchserve and pytorch, and visualize them.

```shell
python tools/torchserve/test_torchserve.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${MODEL_NAME}
[--inference-addr ${INFERENCE_ADDR}] [--result-image ${RESULT_IMAGE}] [--device ${DEVICE}]
```

Example:

```shell
python tools/torchserve/test_torchserve.py \
demo/demo.png \
configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py \
checkpoint/fcn_r50-d8_512x1024_40k_cityscapes_20200604_192608-efe53f0d.pth \
fcn
```
