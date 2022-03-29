# 常见问题解答（FAQ）

我们在这里列出了使用时的一些常见问题及其相应的解决方案。 如果您发现有一些问题被遗漏，请随时提 PR 丰富这个列表。 如果您无法在此获得帮助，请使用 [issue模板](https://github.com/open-mmlab/mmsegmentation/blob/master/.github/ISSUE_TEMPLATE/error-report.md/ )创建问题，但是请在模板中填写所有必填信息，这有助于我们更快定位问题。

## 如何获知模型训练时需要的显卡数量

- 看模型的config文件的命名。可以参考[学习配置文件](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/zh_cn/tutorials/config.md )中的`配置文件命名风格`部分。比如，对于名字为`segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py`的config文件，`8x1`代表训练其对应的模型需要的卡数为8，每张卡中的batch size为1。
- 看模型的log文件。点开该模型的log文件，并在其中搜索`nGPU`，在`nGPU`后的数字个数即训练时所需的卡数。比如，在log文件中搜索`nGPU`得到`nGPU 0,1,2,3,4,5,6,7`的记录，则说明训练该模型需要使用八张卡。
