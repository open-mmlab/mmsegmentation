## Mapillary Vistas Datasets

- The dataset could be download [here](https://www.mapillary.com/dataset/vistas) after registration.

- Mapillary Vistas Dataset use 8-bit with color-palette to store labels. No conversion operation is required.

- Assumption you have put the dataset zip file in `mmsegmentation/data/mapillary`

- Please run the following commands to unzip dataset.

  ```bash
  cd data/mapillary
  unzip An-ZjB1Zm61yAZG0ozTymz8I8NqI4x0MrYrh26dq7kPgfu8vf9ImrdaOAVOFYbJ2pNAgUnVGBmbue9lTgdBOb5BbKXIpFs0fpYWqACbrQDChAA2fdX0zS9PcHu7fY8c-FOvyBVxPNYNFQuM.zip
  ```

- After unzip, you will get Mapillary Vistas Dataset like this structure. Semantic segmentation mask labels in `labels` folder.

  ```none
  mmsegmentation
  ├── mmseg
  ├── tools
  ├── configs
  ├── data
  │   ├── mapillary
  │   │   ├── training
  │   │   │   ├── images
  │   │   │   ├── v1.2
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   └── panoptic
  │   │   │   ├── v2.0
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   ├── panoptic
  |   │   │   │   └── polygons
  │   │   ├── validation
  │   │   │   ├── images
  |   │   │   ├── v1.2
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   └── panoptic
  │   │   │   ├── v2.0
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   ├── panoptic
  |   │   │   │   └── polygons
  ```

- You could set Datasets version with `MapillaryDataset_v1` and `MapillaryDataset_v2` in your configs.
  View the Mapillary Vistas Datasets config file here [V1.2](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/_base_/datasets/mapillary_v1.py) and  [V2.0](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/_base_/datasets/mapillary_v2.py)

- **View datasets labels index and palette**

- **Mapillary Vistas Datasets labels information**
  **v1.2 information**

  ```none
  There are 66 labels classes in v1.2
  0--Bird--[165, 42, 42],
  1--Ground Animal--[0, 192, 0],
  2--Curb--[196, 196, 196],
  3--Fence--[190, 153, 153],
  4--Guard Rail--[180, 165, 180],
  5--Barrier--[90, 120, 150],
  6--Wall--[102, 102, 156],
  7--Bike Lane--[128, 64, 255],
  8--Crosswalk - Plain--[140, 140, 200],
  9--Curb Cut--[170, 170, 170],
  10--Parking--[250, 170, 160],
  11--Pedestrian Area--[96, 96, 96],
  12--Rail Track--[230, 150, 140],
  13--Road--[128, 64, 128],
  14--Service Lane--[110, 110, 110],
  15--Sidewalk--[244, 35, 232],
  16--Bridge--[150, 100, 100],
  17--Building--[70, 70, 70],
  18--Tunnel--[150, 120, 90],
  19--Person--[220, 20, 60],
  20--Bicyclist--[255, 0, 0],
  21--Motorcyclist--[255, 0, 100],
  22--Other Rider--[255, 0, 200],
  23--Lane Marking - Crosswalk--[200, 128, 128],
  24--Lane Marking - General--[255, 255, 255],
  25--Mountain--[64, 170, 64],
  26--Sand--[230, 160, 50],
  27--Sky--[70, 130, 180],
  28--Snow--[190, 255, 255],
  29--Terrain--[152, 251, 152],
  30--Vegetation--[107, 142, 35],
  31--Water--[0, 170, 30],
  32--Banner--[255, 255, 128],
  33--Bench--[250, 0, 30],
  34--Bike Rack--[100, 140, 180],
  35--Billboard--[220, 220, 220],
  36--Catch Basin--[220, 128, 128],
  37--CCTV Camera--[222, 40, 40],
  38--Fire Hydrant--[100, 170, 30],
  39--Junction Box--[40, 40, 40],
  40--Mailbox--[33, 33, 33],
  41--Manhole--[100, 128, 160],
  42--Phone Booth--[142, 0, 0],
  43--Pothole--[70, 100, 150],
  44--Street Light--[210, 170, 100],
  45--Pole--[153, 153, 153],
  46--Traffic Sign Frame--[128, 128, 128],
  47--Utility Pole--[0, 0, 80],
  48--Traffic Light--[250, 170, 30],
  49--Traffic Sign (Back)--[192, 192, 192],
  50--Traffic Sign (Front)--[220, 220, 0],
  51--Trash Can--[140, 140, 20],
  52--Bicycle--[119, 11, 32],
  53--Boat--[150, 0, 255],
  54--Bus--[0, 60, 100],
  55--Car--[0, 0, 142],
  56--Caravan--[0, 0, 90],
  57--Motorcycle--[0, 0, 230],
  58--On Rails--[0, 80, 100],
  59--Other Vehicle--[128, 64, 64],
  60--Trailer--[0, 0, 110],
  61--Truck--[0, 0, 70],
  62--Wheeled Slow--[0, 0, 192],
  63--Car Mount--[32, 32, 32],
  64--Ego Vehicle--[120, 10, 10],
  65--Unlabeled--[0, 0, 0]
  ```

  **v2.0 information**

  ```none
  There are 124 labels classes in v2.0
  0--Bird--[165, 42, 42],
  1--Ground Animal--[0, 192, 0],
  2--Ambiguous Barrier--[250, 170, 31],
  3--Concrete Block--[250, 170, 32],
  4--Curb--[196, 196, 196],
  5--Fence--[190, 153, 153],
  6--Guard Rail--[180, 165, 180],
  7--Barrier--[90, 120, 150],
  8--Road Median--[250, 170, 33],
  9--Road Side--[250, 170, 34],
  10--Lane Separator--[128, 128, 128],
  11--Temporary Barrier--[250, 170, 35],
  12--Wall--[102, 102, 156],
  13--Bike Lane--[128, 64, 255],
  14--Crosswalk - Plain--[140, 140, 200],
  15--Curb Cut--[170, 170, 170],
  16--Driveway--[250, 170, 36],
  17--Parking--[250, 170, 160],
  18--Parking Aisle--[250, 170, 37],
  19--Pedestrian Area--[96, 96, 96],
  20--Rail Track--[230, 150, 140],
  21--Road--[128, 64, 128],
  22--Road Shoulder--[110, 110, 110],
  23--Service Lane--[110, 110, 110],
  24--Sidewalk--[244, 35, 232],
  25--Traffic Island--[128, 196, 128],
  26--Bridge--[150, 100, 100],
  27--Building--[70, 70, 70],
  28--Garage--[150, 150, 150],
  29--Tunnel--[150, 120, 90],
  30--Person--[220, 20, 60],
  31--Person Group--[220, 20, 60],
  32--Bicyclist--[255, 0, 0],
  33--Motorcyclist--[255, 0, 100],
  34--Other Rider--[255, 0, 200],
  35--Lane Marking - Dashed Line--[255, 255, 255],
  36--Lane Marking - Straight Line--[255, 255, 255],
  37--Lane Marking - Zigzag Line--[250, 170, 29],
  38--Lane Marking - Ambiguous--[250, 170, 28],
  39--Lane Marking - Arrow (Left)--[250, 170, 26],
  40--Lane Marking - Arrow (Other)--[250, 170, 25],
  41--Lane Marking - Arrow (Right)--[250, 170, 24],
  42--Lane Marking - Arrow (Split Left or Straight)--[250, 170, 22],
  43--Lane Marking - Arrow (Split Right or Straight)--[250, 170, 21],
  44--Lane Marking - Arrow (Straight)--[250, 170, 20],
  45--Lane Marking - Crosswalk--[255, 255, 255],
  46--Lane Marking - Give Way (Row)--[250, 170, 19],
  47--Lane Marking - Give Way (Single)--[250, 170, 18],
  48--Lane Marking - Hatched (Chevron)--[250, 170, 12],
  49--Lane Marking - Hatched (Diagonal)--[250, 170, 11],
  50--Lane Marking - Other--[255, 255, 255],
  51--Lane Marking - Stop Line--[255, 255, 255],
  52--Lane Marking - Symbol (Bicycle)--[250, 170, 16],
  53--Lane Marking - Symbol (Other)--[250, 170, 15],
  54--Lane Marking - Text--[250, 170, 15],
  55--Lane Marking (only) - Dashed Line--[255, 255, 255],
  56--Lane Marking (only) - Crosswalk--[255, 255, 255],
  57--Lane Marking (only) - Other--[255, 255, 255],
  58--Lane Marking (only) - Test--[255, 255, 255],
  59--Mountain--[64, 170, 64],
  60--Sand--[230, 160, 50],
  61--Sky--[70, 130, 180],
  62--Snow--[190, 255, 255],
  63--Terrain--[152, 251, 152],
  64--Vegetation--[107, 142, 35],
  65--Water--[0, 170, 30],
  66--Banner--[255, 255, 128],
  67--Bench--[250, 0, 30],
  68--Bike Rack--[100, 140, 180],
  69--Catch Basin--[220, 128, 128],
  70--CCTV Camera--[222, 40, 40],
  71--Fire Hydrant--[100, 170, 30],
  72--Junction Box--[40, 40, 40],
  73--Mailbox--[33, 33, 33],
  74--Manhole--[100, 128, 160],
  75--Parking Meter--[20, 20, 255],
  76--Phone Booth--[142, 0, 0],
  77--Pothole--[70, 100, 150],
  78--Signage - Advertisement--[250, 171, 30],
  79--Signage - Ambiguous--[250, 172, 30],
  80--Signage - Back--[250, 173, 30],
  81--Signage - Information--[250, 174, 30],
  82--Signage - Other--[250, 175, 30],
  83--Signage - Store--[250, 176, 30],
  84--Street Light--[210, 170, 100],
  85--Pole--[153, 153, 153],
  86--Pole Group--[153, 153, 153],
  87--Traffic Sign Frame--[128, 128, 128],
  88--Utility Pole--[0, 0, 80],
  89--Traffic Cone--[210, 60, 60],
  90--Traffic Light - General (Single)--[250, 170, 30],
  91--Traffic Light - Pedestrians--[250, 170, 30],
  92--Traffic Light - General (Upright)--[250, 170, 30],
  93--Traffic Light - General (Horizontal)--[250, 170, 30],
  94--Traffic Light - Cyclists--[250, 170, 30],
  95--Traffic Light - Other--[250, 170, 30],
  96--Traffic Sign - Ambiguous--[192, 192, 192],
  97--Traffic Sign (Back)--[192, 192, 192],
  98--Traffic Sign - Direction (Back)--[192, 192, 192],
  99--Traffic Sign - Direction (Front)--[220, 220, 0],
  100--Traffic Sign (Front)--[220, 220, 0],
  101--Traffic Sign - Parking--[0, 0, 196],
  102--Traffic Sign - Temporary (Back)--[192, 192, 192],
  103--Traffic Sign - Temporary (Front)--[220, 220, 0],
  104--Trash Can--[140, 140, 20],
  105--Bicycle--[119, 11, 32],
  106--Boat--[150, 0, 255],
  107--Bus--[0, 60, 100],
  108--Car--[0, 0, 142],
  109--Caravan--[0, 0, 90],
  110--Motorcycle--[0, 0, 230],
  111--On Rails--[0, 80, 100],
  112--Other Vehicle--[128, 64, 64],
  113--Trailer--[0, 0, 110],
  114--Truck--[0, 0, 70],
  115--Vehicle Group--[0, 0, 142],
  116--Wheeled Slow--[0, 0, 192],
  117--Water Valve--[170, 170, 170],
  118--Car Mount--[32, 32, 32],
  119--Dynamic--[111, 74, 0],
  120--Ego Vehicle--[120, 10, 10],
  121--Ground--[81, 0, 81],
  122--Static--[111, 111, 0],
  123--Unlabeled--[0, 0, 0]
  ```
