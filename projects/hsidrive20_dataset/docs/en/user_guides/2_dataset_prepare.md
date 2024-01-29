## HSI Drive 2.0

- You could download HSI Drive 2.0 dataset from [here](https://ipaccess.ehu.eus/HSI-Drive/#download) after just sending an email to gded@ehu.eus with the subject "download HSI-Drive". You will receive a password to uncompress the files.

- After download, unzip by the following instructions:

  ```bash
  7z x -p"password" ./HSI_Drive_v2_0_Phyton.zip

  mv ./HSIDrive20 path_to_mmsegmentation/data
  mv ./HSI_Drive_v2_0_release_notes_Python_version.md path_to_mmsegmentation/data
  mv ./image_numbering.pdf path_to_mmsegmentation/data
  ```

- After unzip, you get

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
