## Changelog

### V0.8 (03/11/2020)

**Highlights**

- Support 4 medical dataset, UNet and CGNet.

**New Features**

- Support customize runner ([#118](https://github.com/open-mmlab/mmsegmentation/pull/118))
- Support UNet ([#161](https://github.com/open-mmlab/mmsegmentation/pull/162))
- Support CHASE_DB1, DRIVE, STARE, HRD ([#203](https://github.com/open-mmlab/mmsegmentation/pull/203))
- Support CGNet ([#223](https://github.com/open-mmlab/mmsegmentation/pull/223))

### V0.7 (07/10/2020)

**Highlights**

- Support Pascal Context dataset and customizing class dataset.

**Bug Fixes**

- Fixed CPU inference ([#153](https://github.com/open-mmlab/mmsegmentation/pull/153))

**New Features**

- Add DeepLab OS16 models ([#154](https://github.com/open-mmlab/mmsegmentation/pull/154))
- Support Pascal Context dataset ([#133](https://github.com/open-mmlab/mmsegmentation/pull/133))
- Support customizing dataset classes ([#71](https://github.com/open-mmlab/mmsegmentation/pull/71))
- Support customizing dataset palette ([#157](https://github.com/open-mmlab/mmsegmentation/pull/157))

**Improvements**

- Support 4D tensor output in ONNX ([#150](https://github.com/open-mmlab/mmsegmentation/pull/150))
- Remove redundancies in ONNX export ([#160](https://github.com/open-mmlab/mmsegmentation/pull/160))
- Migrate to MMCV DepthwiseSeparableConv ([#158](https://github.com/open-mmlab/mmsegmentation/pull/158))
- Migrate to MMCV collect_env ([#137](https://github.com/open-mmlab/mmsegmentation/pull/137))
- Use img_prefix and seg_prefix for loading ([#153](https://github.com/open-mmlab/mmsegmentation/pull/153))

### V0.6 (10/09/2020)

**Highlights**

- Support new methods i.e. MobileNetV2, EMANet, DNL, PointRend, Semantic FPN, Fast-SCNN, ResNeSt.

**Bug Fixes**

- Fixed sliding inference ONNX export ([#90](https://github.com/open-mmlab/mmsegmentation/pull/90))

**New Features**

- Support MobileNet v2 ([#86](https://github.com/open-mmlab/mmsegmentation/pull/86))
- Support EMANet ([#34](https://github.com/open-mmlab/mmsegmentation/pull/34))
- Support DNL ([#37](https://github.com/open-mmlab/mmsegmentation/pull/37))
- Support PointRend ([#109](https://github.com/open-mmlab/mmsegmentation/pull/109))
- Support Semantic FPN ([#94](https://github.com/open-mmlab/mmsegmentation/pull/94))
- Support Fast-SCNN ([#58](https://github.com/open-mmlab/mmsegmentation/pull/58))
- Support ResNeSt backbone ([#47](https://github.com/open-mmlab/mmsegmentation/pull/47))
- Support ONNX export (experimental) ([#12](https://github.com/open-mmlab/mmsegmentation/pull/12))

**Improvements**

- Support Upsample in ONNX ([#100](https://github.com/open-mmlab/mmsegmentation/pull/100))
- Support Windows install (experimental) ([#75](https://github.com/open-mmlab/mmsegmentation/pull/75))
- Add more OCRNet results ([#20](https://github.com/open-mmlab/mmsegmentation/pull/20))
- Add PyTorch 1.6 CI ([#64](https://github.com/open-mmlab/mmsegmentation/pull/64))
- Get version and githash automatically ([#55](https://github.com/open-mmlab/mmsegmentation/pull/55))

### v0.5.1 (11/08/2020)

**Highlights**

- Support FP16 and more generalized OHEM

**Bug Fixes**

- Fixed Pascal VOC conversion script (#19)
- Fixed OHEM weight assign bug (#54)
- Fixed palette type when palette is not given (#27)

**New Features**

- Support FP16 (#21)
- Generalized OHEM (#54)

**Improvements**

- Add load-from flag (#33)
- Fixed training tricks doc about different learning rates of model (#26)
