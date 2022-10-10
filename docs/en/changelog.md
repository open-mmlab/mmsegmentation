## Changelog

### V0.29.0 (10/10/2022)

**New Features**

- Support PoolFormer (CVPR'2022) ([#1537](https://github.com/open-mmlab/mmsegmentation/pull/1537))

**Enhancement**

- Improve structure and readability for FCNHead ([#2142](https://github.com/open-mmlab/mmsegmentation/pull/2142))
- Support IterableDataset in distributed training ([#2151](https://github.com/open-mmlab/mmsegmentation/pull/2151))
- Upgrade .dev scripts ([#2020](https://github.com/open-mmlab/mmsegmentation/pull/2020))
- Upgrade pre-commit hooks ([#2155](https://github.com/open-mmlab/mmsegmentation/pull/2155))

**Bug Fixes**

- Fix mmseg.api.inference inference_segmentor ([#1849](https://github.com/open-mmlab/mmsegmentation/pull/1849))
- fix bug about label_map in evaluation part ([#2075](https://github.com/open-mmlab/mmsegmentation/pull/2075))
- Add missing dependencies to torchserve docker file ([#2133](https://github.com/open-mmlab/mmsegmentation/pull/2133))
- Fix ddp unittest ([#2060](https://github.com/open-mmlab/mmsegmentation/pull/2060))

**Contributors**

- @jinwonkim93 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1849
- @rlatjcj made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2075
- @ShirleyWangCVR made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2151
- @mangelroman made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2133

### V0.28.0 (9/8/2022)

**New Features**

- Support Tversky Loss ([#1896](https://github.com/open-mmlab/mmsegmentation/pull/1986))

**Bug Fixes**

- Fix binary segmentation ([#2016](https://github.com/open-mmlab/mmsegmentation/pull/2016))
- Fix config files ([#1901](https://github.com/open-mmlab/mmsegmentation/pull/1901), [#1893](https://github.com/open-mmlab/mmsegmentation/pull/1893), [#1871](https://github.com/open-mmlab/mmsegmentation/pull/1871))
- Revise documentation ([#1844](https://github.com/open-mmlab/mmsegmentation/pull/1844), [#1980](https://github.com/open-mmlab/mmsegmentation/pull/1980), [#2025](https://github.com/open-mmlab/mmsegmentation/pull/2025), [#1982](https://github.com/open-mmlab/mmsegmentation/pull/1982))
- Fix confusion matrix calculation ([#1992](https://github.com/open-mmlab/mmsegmentation/pull/1992))
- Fix decode head forward_train error ([#1997](https://github.com/open-mmlab/mmsegmentation/pull/1997))

**Contributors**

- @suchot made their first contribution in https://github.com/open-mmlab/mmsegmention/pull/1844
- @TimoK93 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1992

### V0.27.0 (7/28/2022)

**Enhancement**

- Add Swin-L Transformer models ([#1471](https://github.com/open-mmlab/mmsegmentation/pull/1471))
- Update ERFNet results ([#1744](https://github.com/open-mmlab/mmsegmentation/pull/1744))

**Bug Fixes**

- Revise documentation ([#1761](https://github.com/open-mmlab/mmsegmentation/pull/1761), [#1755](https://github.com/open-mmlab/mmsegmentation/pull/1755), [#1802](https://github.com/open-mmlab/mmsegmentation/pull/1802))
- Fix colab tutorial ([#1779](https://github.com/open-mmlab/mmsegmentation/pull/1779))
- Fix segformer checkpoint url ([#1785](https://github.com/open-mmlab/mmsegmentation/pull/1785))

**Contributors**

- @DataSttructure made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1802
- @AkideLiu made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1785
- @mawanda-jun made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1761
- @Yan-Daojiang made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1755

### V0.26.0 (7/1/2022)

**Highlights**

- Update New SegFormer models on ADE20K ([1705](https://github.com/open-mmlab/mmsegmentation/pull/1705))
- Dedicated MMSegWandbHook for MMSegmentation ([1603](https://github.com/open-mmlab/mmsegmentation/pull/1603))

**New Features**

- Update New SegFormer models on ADE20K ([1705](https://github.com/open-mmlab/mmsegmentation/pull/1705))
- Dedicated MMSegWandbHook for MMSegmentation ([1603](https://github.com/open-mmlab/mmsegmentation/pull/1603))
- Add UPerNet r18 results ([1669](https://github.com/open-mmlab/mmsegmentation/pull/1669))

**Enhancement**

- Keep dimension of `cls_token_weight` for easier ONNX deployment ([1642](https://github.com/open-mmlab/mmsegmentation/pull/1642))
- Support infererence with padding ([1607](https://github.com/open-mmlab/mmsegmentation/pull/1607))

**Bug Fixes**

- Fix typos ([#1640](https://github.com/open-mmlab/mmsegmentation/pull/1640), [#1667](https://github.com/open-mmlab/mmsegmentation/pull/1667), [#1656](https://github.com/open-mmlab/mmsegmentation/pull/1656), [#1699](https://github.com/open-mmlab/mmsegmentation/pull/1699), [#1702](https://github.com/open-mmlab/mmsegmentation/pull/1702), [#1695](https://github.com/open-mmlab/mmsegmentation/pull/1695), [#1707](https://github.com/open-mmlab/mmsegmentation/pull/1707), [#1708](https://github.com/open-mmlab/mmsegmentation/pull/1708), [#1721](https://github.com/open-mmlab/mmsegmentation/pull/1721))

**Documentation**

- Fix `mdformat` version to support python3.6 and remove ruby installation ([1672](https://github.com/open-mmlab/mmsegmentation/pull/1672))

**Contributors**

- @RunningLeon made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1642
- @zhouzaida made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1655
- @tkhe made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1667
- @rotorliu made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1656
- @EvelynWang-0423 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1679
- @ZhaoYi1222 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1616
- @Sanster made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1704
- @ayulockin made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1603

### V0.25.0 (6/2/2022)

**Highlights**

- Support PyTorch backend on MLU ([1515](https://github.com/open-mmlab/mmsegmentation/pull/1515))

**Bug Fixes**

- Fix the error of BCE loss when batch size is 1 ([1629](https://github.com/open-mmlab/mmsegmentation/pull/1629))
- Fix bug of `resize` function when align_corners is True ([1592](https://github.com/open-mmlab/mmsegmentation/pull/1592))
- Fix Dockerfile to run demo script in docker container ([1568](https://github.com/open-mmlab/mmsegmentation/pull/1568))
- Correct inference_demo.ipynb path ([1576](https://github.com/open-mmlab/mmsegmentation/pull/1576))
- Fix the `build_segmentor` in colab demo ([1551](https://github.com/open-mmlab/mmsegmentation/pull/1551))
- Fix md2yml script ([1633](https://github.com/open-mmlab/mmsegmentation/pull/1633), [1555](https://github.com/open-mmlab/mmsegmentation/pull/1555))
- Fix main line link in MAE README.md ([1556](https://github.com/open-mmlab/mmsegmentation/pull/1556))
- Fix fastfcn `crop_size` in README.md by ([1597](https://github.com/open-mmlab/mmsegmentation/pull/1597))
- Pip upgrade when testing windows platform ([1610](https://github.com/open-mmlab/mmsegmentation/pull/1610))

**Improvements**

- Delete DS_Store file ([1549](https://github.com/open-mmlab/mmsegmentation/pull/1549))
- Revise owners.yml ([1621](https://github.com/open-mmlab/mmsegmentation/pull/1621), [1534](https://github.com/open-mmlab/mmsegmentation/pull/1543))

**Documentation**

- Rewrite the installation guidance ([1630](https://github.com/open-mmlab/mmsegmentation/pull/1630))
- Format readme ([1635](https://github.com/open-mmlab/mmsegmentation/pull/1635))
- Replace markdownlint with mdformat to avoid ruby installation ([1591](https://github.com/open-mmlab/mmsegmentation/pull/1591))
- Add explanation and usage instructions for data configuration ([1548](https://github.com/open-mmlab/mmsegmentation/pull/1548))
- Configure Myst-parser to parse anchor tag ([1589](https://github.com/open-mmlab/mmsegmentation/pull/1589))
- Update QR code and link for QQ group ([1598](https://github.com/open-mmlab/mmsegmentation/pull/1598), [1574](https://github.com/open-mmlab/mmsegmentation/pull/1574))

**Contributors**

- @atinfinity made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1568
- @DoubleChuang made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1576
- @alpha-baymax made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1515
- @274869388 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1629

### V0.24.1 (5/1/2022)

**Bug Fixes**

- Fix `LayerDecayOptimizerConstructor` for MAE training ([#1539](https://github.com/open-mmlab/mmsegmentation/pull/1539), [#1540](https://github.com/open-mmlab/mmsegmentation/pull/1540))

### V0.24.0 (4/29/2022)

**Highlights**

- Support MAE: Masked Autoencoders Are Scalable Vision Learners
- Support Resnet strikes back

**New Features**

- Support MAE: Masked Autoencoders Are Scalable Vision Learners ([1307](https://github.com/open-mmlab/mmsegmentation/pull/1307), [1523](https://github.com/open-mmlab/mmsegmentation/pull/1523))
- Support Resnet strikes back ([1390](https://github.com/open-mmlab/mmsegmentation/pull/1390))
- Support extra dataloader settings in configs ([1435](https://github.com/open-mmlab/mmsegmentation/pull/1435))

**Bug Fixes**

- Fix input previous results for the last cascade_decode_head ([#1450](https://github.com/open-mmlab/mmsegmentation/pull/1450))
- Fix validation loss logging ([#1494](https://github.com/open-mmlab/mmsegmentation/pull/1494))
- Fix the bug in binary_cross_entropy ([1527](https://github.com/open-mmlab/mmsegmentation/pull/1527))
- Support single channel prediction for Binary Cross Entropy Loss ([#1454](https://github.com/open-mmlab/mmsegmentation/pull/1454))
- Fix potential bugs in accuracy.py ([1496](https://github.com/open-mmlab/mmsegmentation/pull/1496))
- Avoid converting label ids twice by label map during evaluation ([1417](https://github.com/open-mmlab/mmsegmentation/pull/1417))
- Fix bug about label_map ([1445](https://github.com/open-mmlab/mmsegmentation/pull/1445))
- Fix image save path bug in Windows ([1423](https://github.com/open-mmlab/mmsegmentation/pull/1423))
- Fix MMSegmentation Colab demo ([1501](https://github.com/open-mmlab/mmsegmentation/pull/1501), [1452](https://github.com/open-mmlab/mmsegmentation/pull/1452))
- Migrate azure blob for beit checkpoints ([1503](https://github.com/open-mmlab/mmsegmentation/pull/1503))
- Fix bug in `tools/analyse_logs.py` caused by wrong plot_iter in some cases ([1428](https://github.com/open-mmlab/mmsegmentation/pull/1428))

**Improvements**

- Merge BEiT and ConvNext's LR decay optimizer constructors ([#1438](https://github.com/open-mmlab/mmsegmentation/pull/1438))
- Register optimizer constructor with mmseg ([#1456](https://github.com/open-mmlab/mmsegmentation/pull/1456))
- Refactor transformer encode layer in ViT and BEiT backbone ([#1481](https://github.com/open-mmlab/mmsegmentation/pull/1481))
- Add `build_pos_embed` and `build_layers` for BEiT ([1517](https://github.com/open-mmlab/mmsegmentation/pull/1517))
- Add `with_cp` to mit and vit ([1431](https://github.com/open-mmlab/mmsegmentation/pull/1431))
- Fix inconsistent dtype of `seg_label` in stdc decode ([1463](https://github.com/open-mmlab/mmsegmentation/pull/1463))
- Delete random seed for training in `dist_train.sh` ([1519](https://github.com/open-mmlab/mmsegmentation/pull/1519))
- Revise high `workers_per_gpus` in config file ([#1506](https://github.com/open-mmlab/mmsegmentation/pull/1506))
- Add GPG keys and del mmcv version in Dockerfile ([1534](https://github.com/open-mmlab/mmsegmentation/pull/1534))
- Update checkpoint for model in deeplabv3plus ([#1487](https://github.com/open-mmlab/mmsegmentation/pull/1487))
- Add `DistSamplerSeedHook` to set epoch number to dataloader when runner is `EpochBasedRunner` ([1449](https://github.com/open-mmlab/mmsegmentation/pull/1449))
- Provide URLs of Swin Transformer pretrained models ([1389](https://github.com/open-mmlab/mmsegmentation/pull/1389))
- Updating Dockerfiles From Docker Directory and `get_started.md` to reach latest stable version of Python, PyTorch and MMCV ([1446](https://github.com/open-mmlab/mmsegmentation/pull/1446))

**Documentation**

- Add more clearly statement of CPU training/inference ([1518](https://github.com/open-mmlab/mmsegmentation/pull/1518))

**Contributors**

- @jiangyitong made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1431
- @kahkeng made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1447
- @Nourollah made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1446
- @androbaza made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1452
- @Yzichen made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1445
- @whu-pzhang made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1423
- @panfeng-hover made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1417
- @Johnson-Wang made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1496
- @jere357 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1460
- @mfernezir made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1494
- @donglixp made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1503
- @YuanLiuuuuuu made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1307
- @Dawn-bin made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1527

### V0.23.0 (4/1/2022)

**Highlights**

- Support BEiT: BERT Pre-Training of Image Transformers
- Support K-Net: Towards Unified Image Segmentation
- Add `avg_non_ignore` of CELoss to support average loss over non-ignored elements
- Support dataset initialization with file client

**New Features**

- Support BEiT: BERT Pre-Training of Image Transformers ([#1404](https://github.com/open-mmlab/mmsegmentation/pull/1404))
- Support K-Net: Towards Unified Image Segmentation ([#1289](https://github.com/open-mmlab/mmsegmentation/pull/1289))
- Support dataset initialization with file client ([#1402](https://github.com/open-mmlab/mmsegmentation/pull/1402))
- Add class name function for STARE datasets ([#1376](https://github.com/open-mmlab/mmsegmentation/pull/1376))
- Support different seeds on different ranks when distributed training ([#1362](https://github.com/open-mmlab/mmsegmentation/pull/1362))
- Add `nlc2nchw2nlc` and `nchw2nlc2nchw` to simplify tensor with different dimension operation ([#1249](https://github.com/open-mmlab/mmsegmentation/pull/1249))

**Improvements**

- Synchronize random seed for distributed sampler ([#1411](https://github.com/open-mmlab/mmsegmentation/pull/1411))
- Add script and documentation for multi-machine distributed training ([#1383](https://github.com/open-mmlab/mmsegmentation/pull/1383))

**Bug Fixes**

- Add `avg_non_ignore` of CELoss to support average loss over non-ignored elements ([#1409](https://github.com/open-mmlab/mmsegmentation/pull/1409))
- Fix some wrong URLs of models or logs in `./configs` ([#1336](https://github.com/open-mmlab/mmsegmentation/pull/1433))
- Add title and color theme arguments to plot function in `tools/confusion_matrix.py` ([#1401](https://github.com/open-mmlab/mmsegmentation/pull/1401))
- Fix outdated link in Colab demo ([#1392](https://github.com/open-mmlab/mmsegmentation/pull/1392))
- Fix typos ([#1424](https://github.com/open-mmlab/mmsegmentation/pull/1424), [#1405](https://github.com/open-mmlab/mmsegmentation/pull/1405), [#1371](https://github.com/open-mmlab/mmsegmentation/pull/1371), [#1366](https://github.com/open-mmlab/mmsegmentation/pull/1366), [#1363](https://github.com/open-mmlab/mmsegmentation/pull/1363))

**Documentation**

- Add FAQ document ([#1420](https://github.com/open-mmlab/mmsegmentation/pull/1420))
- Fix the config name style description in official docs([#1414](https://github.com/open-mmlab/mmsegmentation/pull/1414))

**Contributors**

- @kinglintianxia made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1371
- @CCODING04 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1376
- @mob5566 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1401
- @xiongnemo made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1392
- @Xiangxu-0103 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1405

### V0.22.1 (3/9/2022)

**Bug Fixes**

- Fix the ZeroDivisionError that all pixels in one image is ignored. ([#1336](https://github.com/open-mmlab/mmsegmentation/pull/1336))

**Improvements**

- Provide URLs of STDC, Segmenter and Twins pretrained models ([#1272](https://github.com/open-mmlab/mmsegmentation/pull/1357))

### V0.22 (3/04/2022)

**Highlights**

- Support ConvNeXt: A ConvNet for the 2020s. Please use the latest MMClassification (0.21.0) to try it out.
- Support iSAID aerial Dataset.
- Officially Support inference on Windows OS.

**New Features**

- Support ConvNeXt: A ConvNet for the 2020s. ([#1216](https://github.com/open-mmlab/mmsegmentation/pull/1216))
- Support iSAID aerial Dataset. ([#1115](https://github.com/open-mmlab/mmsegmentation/pull/1115)
- Generating and plotting confusion matrix. ([#1301](https://github.com/open-mmlab/mmsegmentation/pull/1301))

**Improvements**

- Refactor 4 decoder heads (ASPP, FCN, PSP, UPer): Split forward function into `_forward_feature` and `cls_seg`. ([#1299](https://github.com/open-mmlab/mmsegmentation/pull/1299))
- Add `min_size` arg in `Resize` to keep the shape after resize bigger than slide window. ([#1318](https://github.com/open-mmlab/mmsegmentation/pull/1318))
- Revise pre-commit-hooks. ([#1315](https://github.com/open-mmlab/mmsegmentation/pull/1315))
- Add win-ci. ([#1296](https://github.com/open-mmlab/mmsegmentation/pull/1296))

**Bug Fixes**

- Fix `mlp_ratio` type in Swin Transformer. ([#1274](https://github.com/open-mmlab/mmsegmentation/pull/1274))
- Fix path errors in `./demo` . ([#1269](https://github.com/open-mmlab/mmsegmentation/pull/1269))
- Fix bug in conversion of potsdam. ([#1279](https://github.com/open-mmlab/mmsegmentation/pull/1279))
- Make accuracy take into account `ignore_index`. ([#1259](https://github.com/open-mmlab/mmsegmentation/pull/1259))
- Add Pytorch HardSwish assertion in unit test. ([#1294](https://github.com/open-mmlab/mmsegmentation/pull/1294))
- Fix wrong palette value in vaihingen. ([#1292](https://github.com/open-mmlab/mmsegmentation/pull/1292))
- Fix the bug that SETR cannot load pretrain. ([#1293](https://github.com/open-mmlab/mmsegmentation/pull/1293))
- Update correct `In Collection` in metafile of each configs. ([#1239](https://github.com/open-mmlab/mmsegmentation/pull/1239))
- Upload completed STDC models. ([#1332](https://github.com/open-mmlab/mmsegmentation/pull/1332))
- Fix `DNLHead` exports onnx inference difference type Cast error. ([#1161](https://github.com/open-mmlab/mmsegmentation/pull/1332))

**Contributors**

- @JiaYanhao made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1269
- @andife made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1281
- @SBCV made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1279
- @HJoonKwon made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1259
- @Tsingularity made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1290
- @Waterman0524 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1115
- @MeowZheng made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1315
- @linfangjian01 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1318

### V0.21.1 (2/9/2022)

**Bug Fixes**

- Fix typos in docs. ([#1263](https://github.com/open-mmlab/mmsegmentation/pull/1263))
- Fix repeating log by `setup_multi_processes`. ([#1267](https://github.com/open-mmlab/mmsegmentation/pull/1267))
- Upgrade isort in pre-commit hook. ([#1270](https://github.com/open-mmlab/mmsegmentation/pull/1270))

**Improvements**

- Use MMCV load_state_dict func in ViT/Swin. ([#1272](https://github.com/open-mmlab/mmsegmentation/pull/1272))
- Add exception for PointRend for support CPU-only. ([#1271](https://github.com/open-mmlab/mmsegmentation/pull/1270))

### V0.21 (1/29/2022)

**Highlights**

- Officially Support CPUs training and inference, please use the latest MMCV (1.4.4) to try it out.
- Support Segmenter: Transformer for Semantic Segmentation (ICCV'2021).
- Support ISPRS Potsdam and Vaihingen Dataset.
- Add Mosaic transform and `MultiImageMixDataset` class in `dataset_wrappers`.

**New Features**

- Support Segmenter: Transformer for Semantic Segmentation (ICCV'2021) ([#955](https://github.com/open-mmlab/mmsegmentation/pull/955))
- Support ISPRS Potsdam and Vaihingen Dataset ([#1097](https://github.com/open-mmlab/mmsegmentation/pull/1097), [#1171](https://github.com/open-mmlab/mmsegmentation/pull/1171))
- Add segformer‘s benchmark on cityscapes ([#1155](https://github.com/open-mmlab/mmsegmentation/pull/1155))
- Add auto resume ([#1172](https://github.com/open-mmlab/mmsegmentation/pull/1172))
- Add Mosaic transform and `MultiImageMixDataset` class in `dataset_wrappers` ([#1093](https://github.com/open-mmlab/mmsegmentation/pull/1093), [#1105](https://github.com/open-mmlab/mmsegmentation/pull/1105))
- Add log collector ([#1175](https://github.com/open-mmlab/mmsegmentation/pull/1175))

**Improvements**

- New-style CPU training and inference ([#1251](https://github.com/open-mmlab/mmsegmentation/pull/1251))
- Add UNet benchmark with multiple losses supervision ([#1143](https://github.com/open-mmlab/mmsegmentation/pull/1143))

**Bug Fixes**

- Fix the model statistics in doc for readthedoc ([#1153](https://github.com/open-mmlab/mmsegmentation/pull/1153))
- Set random seed for `palette` if not given ([#1152](https://github.com/open-mmlab/mmsegmentation/pull/1152))
- Add `COCOStuffDataset` in `class_names.py` ([#1222](https://github.com/open-mmlab/mmsegmentation/pull/1222))
- Fix bug in non-distributed multi-gpu training/testing ([#1247](https://github.com/open-mmlab/mmsegmentation/pull/1247))
- Delete unnecessary lines of STDCHead ([#1231](https://github.com/open-mmlab/mmsegmentation/pull/1231))

**Contributors**

- @jbwang1997 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1152
- @BeaverCC made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1206
- @Echo-minn made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1214
- @rstrudel made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/955

### V0.20.2 (12/15/2021)

**Bug Fixes**

- Revise --option to --options to avoid BC-breaking. ([#1140](https://github.com/open-mmlab/mmsegmentation/pull/1140))

### V0.20.1 (12/14/2021)

**Improvements**

- Change options to cfg-options ([#1129](https://github.com/open-mmlab/mmsegmentation/pull/1129))

**Bug Fixes**

- Fix `<!-- [ABSTRACT] -->` in metafile. ([#1127](https://github.com/open-mmlab/mmsegmentation/pull/1127))
- Fix correct `num_classes` of HRNet in `LoveDA` dataset ([#1136](https://github.com/open-mmlab/mmsegmentation/pull/1136))

### V0.20 (12/10/2021)

**Highlights**

- Support Twins ([#989](https://github.com/open-mmlab/mmsegmentation/pull/989))
- Support a real-time segmentation model STDC ([#995](https://github.com/open-mmlab/mmsegmentation/pull/995))
- Support a widely-used segmentation model in lane detection ERFNet ([#960](https://github.com/open-mmlab/mmsegmentation/pull/960))
- Support A Remote Sensing Land-Cover Dataset LoveDA ([#1028](https://github.com/open-mmlab/mmsegmentation/pull/1028))
- Support focal loss ([#1024](https://github.com/open-mmlab/mmsegmentation/pull/1024))

**New Features**

- Support Twins ([#989](https://github.com/open-mmlab/mmsegmentation/pull/989))
- Support a real-time segmentation model STDC ([#995](https://github.com/open-mmlab/mmsegmentation/pull/995))
- Support a widely-used segmentation model in lane detection ERFNet ([#960](https://github.com/open-mmlab/mmsegmentation/pull/960))
- Add SETR cityscapes benchmark ([#1087](https://github.com/open-mmlab/mmsegmentation/pull/1087))
- Add BiSeNetV1 COCO-Stuff 164k benchmark ([#1019](https://github.com/open-mmlab/mmsegmentation/pull/1019))
- Support focal loss ([#1024](https://github.com/open-mmlab/mmsegmentation/pull/1024))
- Add Cutout transform ([#1022](https://github.com/open-mmlab/mmsegmentation/pull/1022))

**Improvements**

- Set a random seed when the user does not set a seed ([#1039](https://github.com/open-mmlab/mmsegmentation/pull/1039))
- Add CircleCI setup ([#1086](https://github.com/open-mmlab/mmsegmentation/pull/1086))
- Skip CI on ignoring given paths ([#1078](https://github.com/open-mmlab/mmsegmentation/pull/1078))
- Add abstract and image for every paper ([#1060](https://github.com/open-mmlab/mmsegmentation/pull/1060))
- Create a symbolic link on windows ([#1090](https://github.com/open-mmlab/mmsegmentation/pull/1090))
- Support video demo using trained model ([#1014](https://github.com/open-mmlab/mmsegmentation/pull/1014))

**Bug Fixes**

- Fix incorrectly loading init_cfg or pretrained models of several transformer models ([#999](https://github.com/open-mmlab/mmsegmentation/pull/999), [#1069](https://github.com/open-mmlab/mmsegmentation/pull/1069), [#1102](https://github.com/open-mmlab/mmsegmentation/pull/1102))
- Fix EfficientMultiheadAttention in SegFormer ([#1037](https://github.com/open-mmlab/mmsegmentation/pull/1037))
- Remove `fp16` folder in `configs` ([#1031](https://github.com/open-mmlab/mmsegmentation/pull/1031))
- Fix several typos in .yml file (Dice Metric [#1041](https://github.com/open-mmlab/mmsegmentation/pull/1041), ADE20K dataset [#1120](https://github.com/open-mmlab/mmsegmentation/pull/1120), Training Memory (GB) [#1083](https://github.com/open-mmlab/mmsegmentation/pull/1083))
- Fix test error when using `--show-dir` ([#1091](https://github.com/open-mmlab/mmsegmentation/pull/1091))
- Fix dist training infinite waiting issue ([#1035](https://github.com/open-mmlab/mmsegmentation/pull/1035))
- Change the upper version of mmcv to 1.5.0 ([#1096](https://github.com/open-mmlab/mmsegmentation/pull/1096))
- Fix symlink failure on Windows ([#1038](https://github.com/open-mmlab/mmsegmentation/pull/1038))
- Cancel previous runs that are not completed ([#1118](https://github.com/open-mmlab/mmsegmentation/pull/1118))
- Unified links of readthedocs in docs ([#1119](https://github.com/open-mmlab/mmsegmentation/pull/1119))

**Contributors**

- @Junjue-Wang made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1028
- @ddebby made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1066
- @del-zhenwu made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1078
- @KangBK0120 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1106
- @zergzzlun made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1091
- @fingertap made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1035
- @irvingzhang0512 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/1014
- @littleSunlxy made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/989
- @lkm2835
- @RockeyCoss
- @MengzhangLI
- @Junjun2016
- @xiexinch
- @xvjiarui

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
