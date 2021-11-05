## Changelog

### V0.19 (11/02/2021)

**Highlights**

- Support TIMMBackbone wrapper ([#998](https://github.com/open-mmlab/mmsegmentation/pull/998))
- Support custom hook ([#428](https://github.com/open-mmlab/mmsegmentation/pull/428))
- Add codespell pre-commit hook ([#920](https://github.com/open-mmlab/mmsegmentation/pull/920))
- Add FastFCN benchmark on ADE20K ([#972](https://github.com/open-mmlab/mmsegmentation/pull/972))

**New Features**

- Support TIMMBackbone wrapper ([#998](https://github.com/open-mmlab/mmsegmentation/pull/998))
- Support custom hook ([#428](https://github.com/open-mmlab/mmsegmentation/pull/428))
- Add FastFCN benchmark on ADE20K ([#972](https://github.com/open-mmlab/mmsegmentation/pull/972))
- Add codespell pre-commit hook and fix typos ([#920](https://github.com/open-mmlab/mmsegmentation/pull/920))

**Improvements**

- Make inputs & channels smaller in unittests ([#1004](https://github.com/open-mmlab/mmsegmentation/pull/1004))
- Change `self.loss_decode` back to `dict` in Single Loss situation ([#1002](https://github.com/open-mmlab/mmsegmentation/pull/1002))

**Bug Fixes**

- Fix typo in usage example ([#1003](https://github.com/open-mmlab/mmsegmentation/pull/1003))
- Add contiguous after permutation in ViT ([#992](https://github.com/open-mmlab/mmsegmentation/pull/992))
- Fix the invalid link ([#985](https://github.com/open-mmlab/mmsegmentation/pull/985))
- Fix bug in CI with python 3.9 ([#994](https://github.com/open-mmlab/mmsegmentation/pull/994))
- Fix bug when loading class name form file in custom dataset ([#923](https://github.com/open-mmlab/mmsegmentation/pull/923))

**Contributors**

- @ShoupingShan made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/923
- @RockeyCoss made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/954
- @HarborYuan made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/992
- @lkm2835 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1003
- @gszh made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/428
- @VVsssssk
- @MengzhangLI
- @Junjun2016

### V0.18 (10/07/2021)

**Highlights**

- Support three real-time segmentation models (ICNet [#884](https://github.com/open-mmlab/mmsegmentation/pull/884), BiSeNetV1 [#851](https://github.com/open-mmlab/mmsegmentation/pull/851), and BiSeNetV2 [#804](https://github.com/open-mmlab/mmsegmentation/pull/804))
- Support one efficient segmentation model (FastFCN [#885](https://github.com/open-mmlab/mmsegmentation/pull/885))
- Support one efficient non-local/self-attention based segmentation model (ISANet [#70](https://github.com/open-mmlab/mmsegmentation/pull/70))
- Support COCO-Stuff 10k and 164k datasets ([#625](https://github.com/open-mmlab/mmsegmentation/pull/625))
- Support evaluate concated dataset separately ([#833](https://github.com/open-mmlab/mmsegmentation/pull/833))
- Support loading GT for evaluation from multi-file backend ([#867](https://github.com/open-mmlab/mmsegmentation/pull/867))

**New Features**

- Support three real-time segmentation models (ICNet [#884](https://github.com/open-mmlab/mmsegmentation/pull/884), BiSeNetV1 [#851](https://github.com/open-mmlab/mmsegmentation/pull/851), and BiSeNetV2 [#804](https://github.com/open-mmlab/mmsegmentation/pull/804))
- Support one efficient segmentation model (FastFCN [#885](https://github.com/open-mmlab/mmsegmentation/pull/885))
- Support one efficient non-local/self-attention based segmentation model (ISANet [#70](https://github.com/open-mmlab/mmsegmentation/pull/70))
- Support COCO-Stuff 10k and 164k datasets ([#625](https://github.com/open-mmlab/mmsegmentation/pull/625))
- Support evaluate concated dataset separately ([#833](https://github.com/open-mmlab/mmsegmentation/pull/833))

**Improvements**

- Support loading GT for evaluation from multi-file backend ([#867](https://github.com/open-mmlab/mmsegmentation/pull/867))
- Auto-convert SyncBN to BN when training on DP automatly([#772](https://github.com/open-mmlab/mmsegmentation/pull/772))
- Refactor Swin-Transformer ([#800](https://github.com/open-mmlab/mmsegmentation/pull/800))

**Bug Fixes**

- Update mmcv installation in dockerfile ([#860](https://github.com/open-mmlab/mmsegmentation/pull/860))
- Fix number of iteration bug when resuming checkpoint in distributed train ([#866](https://github.com/open-mmlab/mmsegmentation/pull/866))
- Fix parsing parse in val_step ([#906](https://github.com/open-mmlab/mmsegmentation/pull/906))

### V0.17 (09/01/2021)

**Highlights**

- Support SegFormer
- Support DPT
- Support Dark Zurich and Nighttime Driving datasets
- Support progressive evaluation

**New Features**

- Support SegFormer ([#599](https://github.com/open-mmlab/mmsegmentation/pull/599))
- Support DPT ([#605](https://github.com/open-mmlab/mmsegmentation/pull/605))
- Support Dark Zurich and Nighttime Driving datasets ([#815](https://github.com/open-mmlab/mmsegmentation/pull/815))
- Support progressive evaluation ([#709](https://github.com/open-mmlab/mmsegmentation/pull/709))

**Improvements**

- Add multiscale_output interface and unittests for HRNet ([#830](https://github.com/open-mmlab/mmsegmentation/pull/830))
- Support inherit cityscapes dataset ([#750](https://github.com/open-mmlab/mmsegmentation/pull/750))
- Fix some typos in README.md ([#824](https://github.com/open-mmlab/mmsegmentation/pull/824))
- Delete convert function and add instruction to ViT/Swin README.md ([#791](https://github.com/open-mmlab/mmsegmentation/pull/791))
- Add vit/swin/mit convert weight scripts ([#783](https://github.com/open-mmlab/mmsegmentation/pull/783))
- Add copyright files ([#796](https://github.com/open-mmlab/mmsegmentation/pull/796))

**Bug Fixes**

- Fix invalid checkpoint link in inference_demo.ipynb ([#814](https://github.com/open-mmlab/mmsegmentation/pull/814))
- Ensure that items in dataset have the same order across multi machine ([#780](https://github.com/open-mmlab/mmsegmentation/pull/780))
- Fix the log error ([#766](https://github.com/open-mmlab/mmsegmentation/pull/766))

### V0.16 (08/04/2021)

**Highlights**

- Support PyTorch 1.9
- Support SegFormer backbone MiT
- Support md2yml pre-commit hook
- Support frozen stage for HRNet

**New Features**

- Support SegFormer backbone MiT ([#594](https://github.com/open-mmlab/mmsegmentation/pull/594))
- Support md2yml pre-commit hook ([#732](https://github.com/open-mmlab/mmsegmentation/pull/732))
- Support mim ([#717](https://github.com/open-mmlab/mmsegmentation/pull/717))
- Add mmseg2torchserve tool ([#552](https://github.com/open-mmlab/mmsegmentation/pull/552))

**Improvements**

- Support hrnet frozen stage ([#743](https://github.com/open-mmlab/mmsegmentation/pull/743))
- Add template of reimplementation questions ([#741](https://github.com/open-mmlab/mmsegmentation/pull/741))
- Output pdf and epub formats for readthedocs ([#742](https://github.com/open-mmlab/mmsegmentation/pull/742))
- Refine the docstring of ResNet ([#723](https://github.com/open-mmlab/mmsegmentation/pull/723))
- Replace interpolate with resize ([#731](https://github.com/open-mmlab/mmsegmentation/pull/731))
- Update resource limit ([#700](https://github.com/open-mmlab/mmsegmentation/pull/700))
- Update config.md ([#678](https://github.com/open-mmlab/mmsegmentation/pull/678))

**Bug Fixes**

- Fix ATTENTION registry ([#729](https://github.com/open-mmlab/mmsegmentation/pull/729))
- Fix analyze log script ([#716](https://github.com/open-mmlab/mmsegmentation/pull/716))
- Fix doc api display ([#725](https://github.com/open-mmlab/mmsegmentation/pull/725))
- Fix patch_embed and pos_embed mismatch error ([#685](https://github.com/open-mmlab/mmsegmentation/pull/685))
- Fix efficient test for multi-node ([#707](https://github.com/open-mmlab/mmsegmentation/pull/707))
- Fix init_cfg in resnet backbone ([#697](https://github.com/open-mmlab/mmsegmentation/pull/697))
- Fix efficient test bug ([#702](https://github.com/open-mmlab/mmsegmentation/pull/702))
- Fix url error in config docs ([#680](https://github.com/open-mmlab/mmsegmentation/pull/680))
- Fix mmcv installation ([#676](https://github.com/open-mmlab/mmsegmentation/pull/676))
- Fix torch version ([#670](https://github.com/open-mmlab/mmsegmentation/pull/670))

**Contributors**

@sshuair @xiexinch @Junjun2016 @mmeendez8 @xvjiarui @sennnnn @puhsu @BIGWangYuDong @keke1u @daavoo

### V0.15 (07/04/2021)

**Highlights**

- Support ViT, SETR, and Swin-Transformer
- Add Chinese documentation
- Unified parameter initialization

**Bug Fixes**

- Fix typo and links ([#608](https://github.com/open-mmlab/mmsegmentation/pull/608))
- Fix Dockerfile ([#607](https://github.com/open-mmlab/mmsegmentation/pull/607))
- Fix ViT init ([#609](https://github.com/open-mmlab/mmsegmentation/pull/609))
- Fix mmcv version compatible table ([#658](https://github.com/open-mmlab/mmsegmentation/pull/658))
- Fix model links of DMNEt ([#660](https://github.com/open-mmlab/mmsegmentation/pull/660))

**New Features**

- Support loading DeiT weights ([#538](https://github.com/open-mmlab/mmsegmentation/pull/538))
- Support SETR ([#531](https://github.com/open-mmlab/mmsegmentation/pull/531), [#635](https://github.com/open-mmlab/mmsegmentation/pull/635))
- Add config and models for ViT backbone with UperHead ([#520](https://github.com/open-mmlab/mmsegmentation/pull/531), [#635](https://github.com/open-mmlab/mmsegmentation/pull/520))
- Support Swin-Transformer ([#511](https://github.com/open-mmlab/mmsegmentation/pull/511))
- Add higher accuracy FastSCNN ([#606](https://github.com/open-mmlab/mmsegmentation/pull/606))
- Add Chinese documentation ([#666](https://github.com/open-mmlab/mmsegmentation/pull/666))

**Improvements**

- Unified parameter initialization ([#567](https://github.com/open-mmlab/mmsegmentation/pull/567))
- Separate CUDA and CPU in  github action CI ([#602](https://github.com/open-mmlab/mmsegmentation/pull/602))
- Support persistent dataloader worker ([#646](https://github.com/open-mmlab/mmsegmentation/pull/646))
- Update meta file fields ([#661](https://github.com/open-mmlab/mmsegmentation/pull/661), [#664](https://github.com/open-mmlab/mmsegmentation/pull/664))

### V0.14 (06/02/2021)

**Highlights**

- Support ONNX to TensorRT
- Support MIM

**Bug Fixes**

- Fix ONNX to TensorRT verify ([#547](https://github.com/open-mmlab/mmsegmentation/pull/547))
- Fix save best for EvalHook ([#575](https://github.com/open-mmlab/mmsegmentation/pull/575))

**New Features**

- Support loading DeiT weights ([#538](https://github.com/open-mmlab/mmsegmentation/pull/538))
- Support ONNX to TensorRT ([#542](https://github.com/open-mmlab/mmsegmentation/pull/542))
- Support output results for ADE20k ([#544](https://github.com/open-mmlab/mmsegmentation/pull/544))
- Support MIM ([#549](https://github.com/open-mmlab/mmsegmentation/pull/549))

**Improvements**

- Add option for ViT output shape ([#530](https://github.com/open-mmlab/mmsegmentation/pull/530))
- Infer batch size using len(result) ([#532](https://github.com/open-mmlab/mmsegmentation/pull/532))
- Add compatible table between MMSeg and MMCV ([#558](https://github.com/open-mmlab/mmsegmentation/pull/558))

### V0.13 (05/05/2021)

**Highlights**

- Support Pascal Context Class-59 dataset.
- Support Visual Transformer Backbone.
- Support mFscore metric.

**Bug Fixes**

- Fixed Colaboratory tutorial ([#451](https://github.com/open-mmlab/mmsegmentation/pull/451))
- Fixed mIoU calculation range ([#471](https://github.com/open-mmlab/mmsegmentation/pull/471))
- Fixed sem_fpn, unet README.md ([#492](https://github.com/open-mmlab/mmsegmentation/pull/492))
- Fixed `num_classes` in FCN for Pascal Context 60-class dataset ([#488](https://github.com/open-mmlab/mmsegmentation/pull/488))
- Fixed FP16 inference ([#497](https://github.com/open-mmlab/mmsegmentation/pull/497))

**New Features**

- Support dynamic export and visualize to pytorch2onnx ([#463](https://github.com/open-mmlab/mmsegmentation/pull/463))
- Support export to torchscript ([#469](https://github.com/open-mmlab/mmsegmentation/pull/469), [#499](https://github.com/open-mmlab/mmsegmentation/pull/499))
- Support Pascal Context Class-59 dataset ([#459](https://github.com/open-mmlab/mmsegmentation/pull/459))
- Support Visual Transformer backbone ([#465](https://github.com/open-mmlab/mmsegmentation/pull/465))
- Support UpSample Neck ([#512](https://github.com/open-mmlab/mmsegmentation/pull/512))
- Support mFscore metric ([#509](https://github.com/open-mmlab/mmsegmentation/pull/509))

**Improvements**

- Add more CI for PyTorch ([#460](https://github.com/open-mmlab/mmsegmentation/pull/460))
- Add print model graph args for tools/print_config.py ([#451](https://github.com/open-mmlab/mmsegmentation/pull/451))
- Add cfg links in modelzoo README.md ([#468](https://github.com/open-mmlab/mmsegmentation/pull/469))
- Add BaseSegmentor import to segmentors/__init__.py ([#495](https://github.com/open-mmlab/mmsegmentation/pull/495))
- Add MMOCR, MMGeneration links ([#501](https://github.com/open-mmlab/mmsegmentation/pull/501), [#506](https://github.com/open-mmlab/mmsegmentation/pull/506))
- Add Chinese QR code ([#506](https://github.com/open-mmlab/mmsegmentation/pull/506))
- Use MMCV MODEL_REGISTRY ([#515](https://github.com/open-mmlab/mmsegmentation/pull/515))
- Add ONNX testing tools ([#498](https://github.com/open-mmlab/mmsegmentation/pull/498))
- Replace data_dict calling 'img' key to support MMDet3D ([#514](https://github.com/open-mmlab/mmsegmentation/pull/514))
- Support reading class_weight from file in loss function ([#513](https://github.com/open-mmlab/mmsegmentation/pull/513))
- Make tags as comment ([#505](https://github.com/open-mmlab/mmsegmentation/pull/505))
- Use MMCV EvalHook ([#438](https://github.com/open-mmlab/mmsegmentation/pull/438))

### V0.12 (04/03/2021)

**Highlights**

- Support FCN-Dilate 6 model.
- Support Dice Loss.

**Bug Fixes**

- Fixed PhotoMetricDistortion Doc ([#388](https://github.com/open-mmlab/mmsegmentation/pull/388))
- Fixed install scripts ([#399](https://github.com/open-mmlab/mmsegmentation/pull/399))
- Fixed Dice Loss multi-class ([#417](https://github.com/open-mmlab/mmsegmentation/pull/417))

**New Features**

- Support Dice Loss ([#396](https://github.com/open-mmlab/mmsegmentation/pull/396))
- Add plot logs tool ([#426](https://github.com/open-mmlab/mmsegmentation/pull/426))
- Add opacity option to show_result ([#425](https://github.com/open-mmlab/mmsegmentation/pull/425))
- Speed up mIoU metric ([#430](https://github.com/open-mmlab/mmsegmentation/pull/430))

**Improvements**

- Refactor unittest file structure ([#440](https://github.com/open-mmlab/mmsegmentation/pull/440))
- Fix typos in the repo ([#449](https://github.com/open-mmlab/mmsegmentation/pull/449))
- Include class-level metrics in the log ([#445](https://github.com/open-mmlab/mmsegmentation/pull/445))

### V0.11 (02/02/2021)

**Highlights**

- Support memory efficient test, add more UNet models.

**Bug Fixes**

- Fixed TTA resize scale ([#334](https://github.com/open-mmlab/mmsegmentation/pull/334))
- Fixed CI for pip 20.3 ([#307](https://github.com/open-mmlab/mmsegmentation/pull/307))
- Fixed ADE20k test ([#359](https://github.com/open-mmlab/mmsegmentation/pull/359))

**New Features**

- Support memory efficient test ([#330](https://github.com/open-mmlab/mmsegmentation/pull/330))
- Add more UNet benchmarks ([#324](https://github.com/open-mmlab/mmsegmentation/pull/324))
- Support Lovasz Loss ([#351](https://github.com/open-mmlab/mmsegmentation/pull/351))

**Improvements**

- Move train_cfg/test_cfg inside model ([#341](https://github.com/open-mmlab/mmsegmentation/pull/341))

### V0.10 (01/01/2021)

**Highlights**

- Support MobileNetV3, DMNet, APCNet. Add models of ResNet18V1b, ResNet18V1c, ResNet50V1b.

**Bug Fixes**

- Fixed CPU TTA ([#276](https://github.com/open-mmlab/mmsegmentation/pull/276))
- Fixed CI for pip 20.3 ([#307](https://github.com/open-mmlab/mmsegmentation/pull/307))

**New Features**

- Add ResNet18V1b, ResNet18V1c, ResNet50V1b, ResNet101V1b models ([#316](https://github.com/open-mmlab/mmsegmentation/pull/316))
- Support MobileNetV3 ([#268](https://github.com/open-mmlab/mmsegmentation/pull/268))
- Add 4 retinal vessel segmentation benchmark  ([#315](https://github.com/open-mmlab/mmsegmentation/pull/315))
- Support DMNet ([#313](https://github.com/open-mmlab/mmsegmentation/pull/313))
- Support APCNet ([#299](https://github.com/open-mmlab/mmsegmentation/pull/299))

**Improvements**

- Refactor Documentation page ([#311](https://github.com/open-mmlab/mmsegmentation/pull/311))
- Support resize data augmentation according to original image size ([#291](https://github.com/open-mmlab/mmsegmentation/pull/291))

### V0.9 (30/11/2020)

**Highlights**

- Support 4 medical dataset, UNet and CGNet.

**New Features**

- Support RandomRotate transform ([#215](https://github.com/open-mmlab/mmsegmentation/pull/215), [#260](https://github.com/open-mmlab/mmsegmentation/pull/260))
- Support RGB2Gray transform ([#227](https://github.com/open-mmlab/mmsegmentation/pull/227))
- Support Rerange transform ([#228](https://github.com/open-mmlab/mmsegmentation/pull/228))
- Support ignore_index for BCE loss ([#210](https://github.com/open-mmlab/mmsegmentation/pull/210))
- Add modelzoo statistics ([#263](https://github.com/open-mmlab/mmsegmentation/pull/263))
- Support Dice evaluation metric ([#225](https://github.com/open-mmlab/mmsegmentation/pull/225))
- Support Adjust Gamma transform ([#232](https://github.com/open-mmlab/mmsegmentation/pull/232))
- Support CLAHE transform ([#229](https://github.com/open-mmlab/mmsegmentation/pull/229))

**Bug Fixes**

- Fixed detail API link ([#267](https://github.com/open-mmlab/mmsegmentation/pull/267))

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
