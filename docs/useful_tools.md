Apart from training/testing scripts, We provide lots of useful tools under the
 `tools/` directory.

### Get the FLOPs and params (experimental)

We provide a script adapted from [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) to compute the FLOPs and params of a given model.

```shell
python tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

You will get the result like this.

```none
==============================
Input shape: (3, 2048, 1024)
Flops: 1429.68 GMac
Params: 48.98 M
==============================
```

**Note**: This tool is still experimental and we do not guarantee that the number is correct. You may well use the result for simple comparisons, but double check it before you adopt it in technical reports or papers.

(1) FLOPs are related to the input shape while parameters are not. The default input shape is (1, 3, 1280, 800).
(2) Some operators are not counted into FLOPs like GN and custom operators.

### Publish a model

Before you upload a model to AWS, you may want to
(1) convert model weights to CPU tensors, (2) delete the optimizer states and
(3) compute the hash of the checkpoint file and append the hash id to the filename.

```shell
python tools/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/publish_model.py work_dirs/pspnet/latest.pth psp_r50_hszhao_200ep.pth
```

The final output filename will be `psp_r50_512x1024_40ki_cityscapes-{hash id}.pth`.

### Convert to ONNX (experimental)

We provide a script to convert model to [ONNX](https://github.com/onnx/onnx) format. The converted model could be visualized by tools like [Netron](https://github.com/lutzroeder/netron). Besides, we also support comparing the output results between Pytorch and ONNX model.

```shell
python tools/pytorch2onnx.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} --output-file ${ONNX_FILE} [--shape ${INPUT_SHAPE} --verify]
```

**Note**: This tool is still experimental. Some customized operators are not supported for now.

## Miscellaneous

### Print the entire config

`tools/print_config.py` prints the whole config verbatim, expanding all its
 imports.

```shell
python tools/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
```
