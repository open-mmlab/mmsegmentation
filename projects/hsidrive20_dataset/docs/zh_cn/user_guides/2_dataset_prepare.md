## HSI Drive 2.0

- 您可以从以下位置下载 HSI Drive 2.0 数据集 [here](https://ipaccess.ehu.eus/HSI-Drive/#download) 刚刚向 gded@ehu.eus 发送主题为“下载 HSI-Drive”的电子邮件后 您将收到解压缩文件的密码.

- 下载后，按照以下说明解压：

  ```bash
  7z x -p"password" ./HSI_Drive_v2_0_Phyton.zip

  mv ./HSIDrive20 path_to_mmsegmentation/data
  mv ./HSI_Drive_v2_0_release_notes_Python_version.md path_to_mmsegmentation/data
  mv ./image_numbering.pdf path_to_mmsegmentation/data
  ```

- 解压后得到:

```none
mmsegmentation
├── mmseg
├── tools
├── configs
├── data
│   ├── HSIDrive20
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── test
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── test
│   │   ├── images_MF
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── test
│   │   ├── RGB
│   │   ├── training_filenames.txt
│   │   ├── validation_filenames.txt
│   │   ├── test_filenames.txt
│   ├── HSI_Drive_v2_0_release_notes_Python_version.md
│   ├── image_numbering.pdf
```
