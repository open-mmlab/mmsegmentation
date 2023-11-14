codebase_config= dict(
    type="mmseg",
    task="Segmentation",
)

backend_config = dict(
    type="onnxruntime"
)

onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    # controll the operator set for jetson tensorrt compatibility
    # https://github.com/onnx/onnx/blob/main/docs/Operators.md
    # https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md
    opset_version=11, 
    save_file="model.onnx",
    input_names=['input'],
    output_names=['output'],
    input_shape=None,
)


