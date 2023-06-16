# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from mmseg.core import intersect_and_union
from mmseg.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from .builder import DATASETS, PIPELINES
from .custom import CustomDataset


@PIPELINES.register_module()
class LoadImageNetSImageFromFile(LoadImageFromFile):
    """Load an image from the ImageNetS dataset.

    To avoid out of memory, images that are too large will
    be downsampled to the scale of 1000.

    Args:
        downsample_large_image (bool): Whether to downsample the large images.
            False may cause out of memory.
            Defaults to True.
    """

    def __init__(self, downsample_large_image=True, **kwargs):
        super().__init__(**kwargs)
        self.downsample_large_image = downsample_large_image

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        results = super().__call__(results)
        if not self.downsample_large_image:
            return results

        # Images that are too large
        # (H * W > 1000 * 100,
        # these images are included in ImageNetSDataset.LARGES)
        # will be downsampled to 1000 along the longer side.
        H, W = results['img_shape'][:2]
        if H * W > pow(1000, 2):
            if H > W:
                target_size = (int(1000 * W / H), 1000)
            else:
                target_size = (1000, int(1000 * H / W))

            results['img'] = mmcv.imresize(
                results['img'], size=target_size, interpolation='bilinear')
            if self.to_float32:
                results['img'] = results['img'].astype(np.float32)

            results['img_shape'] = results['img'].shape
            results['ori_shape'] = results['img'].shape
            # Set initial values for default meta_keys
            results['pad_shape'] = results['img'].shape
        return results


@PIPELINES.register_module()
class LoadImageNetSAnnotations(LoadAnnotations):
    """Load annotations for the ImageNetS dataset. The annotations in
    ImageNet-S are saved as RGB images.

    The annotations with format of RGB should be
    converted to the format of Gray as R + G * 256.
    """

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        results = super().__call__(results)

        # The annotations in ImageNet-S are saved as RGB images,
        # due to 919 > 255 (upper bound of gray images).

        # For training,
        # the annotations with format of RGB should be
        # converted to the format of Gray as R + G * 256.
        results['gt_semantic_seg'] = \
            results['gt_semantic_seg'][:, :, 1] * 256 + \
            results['gt_semantic_seg'][:, :, 2]
        results['gt_semantic_seg'] = results['gt_semantic_seg'].astype(
            np.int32)
        return results


@DATASETS.register_module()
class ImageNetSDataset(CustomDataset):
    """ImageNet-S dataset.

    In segmentation map annotation for ImageNet-S, 0 stands for others, which
    is not included in 50/300/919 categories. ``ignore_index`` is fixed to
    1000. The ``img_suffix`` is fixed to '.JPEG' and ``seg_map_suffix`` is
    fixed to '.png'.
    """
    CLASSES50 = ('others', 'goldfish', 'tiger shark', 'goldfinch', 'tree frog',
                 'kuvasz', 'red fox', 'siamese cat', 'american black bear',
                 'ladybug', 'sulphur butterfly', 'wood rabbit', 'hamster',
                 'wild boar', 'gibbon', 'african elephant', 'giant panda',
                 'airliner', 'ashcan', 'ballpoint', 'beach wagon', 'boathouse',
                 'bullet train', 'cellular telephone', 'chest', 'clog',
                 'container ship', 'digital watch', 'dining table',
                 'golf ball', 'grand piano', 'iron', 'lab coat', 'mixing bowl',
                 'motor scooter', 'padlock', 'park bench', 'purse',
                 'streetcar', 'table lamp', 'television', 'toilet seat',
                 'umbrella', 'vase', 'water bottle', 'water tower', 'yawl',
                 'street sign', 'lemon', 'carbonara', 'agaric')
    CLASSES300 = (
        'others', 'tench', 'goldfish', 'tiger shark', 'hammerhead',
        'electric ray', 'ostrich', 'goldfinch', 'house finch',
        'indigo bunting', 'kite', 'common newt', 'axolotl', 'tree frog',
        'tailed frog', 'mud turtle', 'banded gecko', 'american chameleon',
        'whiptail', 'african chameleon', 'komodo dragon', 'american alligator',
        'triceratops', 'thunder snake', 'ringneck snake', 'king snake',
        'rock python', 'horned viper', 'harvestman', 'scorpion',
        'garden spider', 'tick', 'african grey', 'lorikeet',
        'red-breasted merganser', 'wallaby', 'koala', 'jellyfish',
        'sea anemone', 'conch', 'fiddler crab', 'american lobster',
        'spiny lobster', 'isopod', 'bittern', 'crane', 'limpkin', 'bustard',
        'albatross', 'toy terrier', 'afghan hound', 'bluetick', 'borzoi',
        'irish wolfhound', 'whippet', 'ibizan hound', 'staffordshire '
        'bullterrier', 'border terrier', 'yorkshire terrier',
        'lakeland terrier', 'giant schnauzer', 'standard schnauzer',
        'scotch terrier', 'lhasa', 'english setter', 'clumber',
        'english springer', 'welsh springer spaniel', 'kuvasz', 'kelpie',
        'doberman', 'miniature pinscher', 'malamute', 'pug', 'leonberg',
        'great pyrenees', 'samoyed', 'brabancon griffon', 'cardigan', 'coyote',
        'red fox', 'kit fox', 'grey fox', 'persian cat', 'siamese cat',
        'cougar', 'lynx', 'tiger', 'american black bear', 'sloth bear',
        'ladybug', 'leaf beetle', 'weevil', 'bee', 'cicada', 'leafhopper',
        'damselfly', 'ringlet', 'cabbage butterfly', 'sulphur butterfly',
        'sea cucumber', 'wood rabbit', 'hare', 'hamster', 'wild boar',
        'hippopotamus', 'bighorn', 'ibex', 'badger', 'three-toed sloth',
        'orangutan', 'gibbon', 'colobus', 'spider monkey', 'squirrel monkey',
        'madagascar cat', 'indian elephant', 'african elephant', 'giant panda',
        'barracouta', 'eel', 'coho', 'academic gown', 'accordion', 'airliner',
        'ambulance', 'analog clock', 'ashcan', 'backpack', 'balloon',
        'ballpoint', 'barbell', 'barn', 'bassoon', 'bath towel', 'beach wagon',
        'bicycle-built-for-two', 'binoculars', 'boathouse', 'bonnet',
        'bookcase', 'bow', 'brass', 'breastplate', 'bullet train', 'cannon',
        'can opener', "carpenter's kit", 'cassette', 'cellular telephone',
        'chain saw', 'chest', 'china cabinet', 'clog', 'combination lock',
        'container ship', 'corkscrew', 'crate', 'crock pot', 'digital watch',
        'dining table', 'dishwasher', 'doormat', 'dutch oven', 'electric fan',
        'electric locomotive', 'envelope', 'file', 'folding chair',
        'football helmet', 'freight car', 'french horn', 'fur coat',
        'garbage truck', 'goblet', 'golf ball', 'grand piano', 'half track',
        'hamper', 'hard disc', 'harmonica', 'harvester', 'hook',
        'horizontal bar', 'horse cart', 'iron', "jack-o'-lantern", 'lab coat',
        'ladle', 'letter opener', 'liner', 'mailbox', 'megalith',
        'military uniform', 'milk can', 'mixing bowl', 'monastery', 'mortar',
        'mosquito net', 'motor scooter', 'mountain bike', 'mountain tent',
        'mousetrap', 'necklace', 'nipple', 'ocarina', 'padlock', 'palace',
        'parallel bars', 'park bench', 'pedestal', 'pencil sharpener',
        'pickelhaube', 'pillow', 'planetarium', 'plastic bag',
        'polaroid camera', 'pole', 'pot', 'purse', 'quilt', 'radiator',
        'radio', 'radio telescope', 'rain barrel', 'reflex camera',
        'refrigerator', 'rifle', 'rocking chair', 'rubber eraser', 'rule',
        'running shoe', 'sewing machine', 'shield', 'shoji', 'ski', 'ski mask',
        'slot', 'soap dispenser', 'soccer ball', 'sock', 'soup bowl',
        'space heater', 'spider web', 'spindle', 'sports car',
        'steel arch bridge', 'stethoscope', 'streetcar', 'submarine',
        'swimming trunks', 'syringe', 'table lamp', 'tank', 'teddy',
        'television', 'throne', 'tile roof', 'toilet seat', 'trench coat',
        'trimaran', 'typewriter keyboard', 'umbrella', 'vase', 'volleyball',
        'wardrobe', 'warplane', 'washer', 'water bottle', 'water tower',
        'whiskey jug', 'wig', 'wine bottle', 'wok', 'wreck', 'yawl', 'yurt',
        'street sign', 'traffic light', 'consomme', 'ice cream', 'bagel',
        'cheeseburger', 'hotdog', 'mashed potato', 'spaghetti squash',
        'bell pepper', 'cardoon', 'granny smith', 'strawberry', 'lemon',
        'carbonara', 'burrito', 'cup', 'coral reef', "yellow lady's slipper",
        'buckeye', 'agaric', 'gyromitra', 'earthstar', 'bolete')
    CLASSES919 = (
        'others', 'house finch', 'stupa', 'agaric', 'hen-of-the-woods',
        'wild boar', 'kit fox', 'desk', 'beaker', 'spindle', 'lipstick',
        'cardoon', 'ringneck snake', 'daisy', 'sturgeon', 'scorpion',
        'pelican', 'bustard', 'rock crab', 'rock beauty', 'minivan', 'menu',
        'thunder snake', 'zebra', 'partridge', 'lacewing', 'starfish',
        'italian greyhound', 'marmot', 'cardigan', 'plate', 'ballpoint',
        'chesapeake bay retriever', 'pirate', 'potpie', 'keeshond', 'dhole',
        'waffle iron', 'cab', 'american egret', 'colobus', 'radio telescope',
        'gordon setter', 'mousetrap', 'overskirt', 'hamster', 'wine bottle',
        'bluetick', 'macaque', 'bullfrog', 'junco', 'tusker', 'scuba diver',
        'pool table', 'samoyed', 'mailbox', 'purse', 'monastery', 'bathtub',
        'window screen', 'african crocodile', 'traffic light', 'tow truck',
        'radio', 'recreational vehicle', 'grey whale', 'crayfish',
        'rottweiler', 'racer', 'whistle', 'pencil box', 'barometer',
        'cabbage butterfly', 'sloth bear', 'rhinoceros beetle', 'guillotine',
        'rocking chair', 'sports car', 'bouvier des flandres', 'border collie',
        'fiddler crab', 'slot', 'go-kart', 'cocker spaniel', 'plate rack',
        'common newt', 'tile roof', 'marimba', 'moped', 'terrapin', 'oxcart',
        'lionfish', 'bassinet', 'rain barrel', 'american black bear', 'goose',
        'half track', 'kite', 'microphone', 'shield', 'mexican hairless',
        'measuring cup', 'bubble', 'platypus', 'saint bernard', 'police van',
        'vase', 'lhasa', 'wardrobe', 'teapot', 'hummingbird', 'revolver',
        'jinrikisha', 'mailbag', 'red-breasted merganser', 'assault rifle',
        'loudspeaker', 'fig', 'american lobster', 'can opener', 'arctic fox',
        'broccoli', 'long-horned beetle', 'television', 'airship',
        'black stork', 'marmoset', 'panpipe', 'drumstick', 'knee pad',
        'lotion', 'french loaf', 'throne', 'jeep', 'jersey', 'tiger cat',
        'cliff', 'sealyham terrier', 'strawberry', 'minibus', 'goldfinch',
        'goblet', 'burrito', 'harp', 'tractor', 'cornet', 'leopard', 'fly',
        'fireboat', 'bolete', 'barber chair', 'consomme', 'tripod',
        'breastplate', 'pineapple', 'wok', 'totem pole', 'alligator lizard',
        'common iguana', 'digital clock', 'bighorn', 'siamese cat', 'bobsled',
        'irish setter', 'zucchini', 'crock pot', 'loggerhead',
        'irish wolfhound', 'nipple', 'rubber eraser', 'impala', 'barbell',
        'snow leopard', 'siberian husky', 'necklace', 'manhole cover',
        'electric fan', 'hippopotamus', 'entlebucher', 'prison', 'doberman',
        'ruffed grouse', 'coyote', 'toaster', 'puffer', 'black swan',
        'schipperke', 'file', 'prairie chicken', 'hourglass',
        'greater swiss mountain dog', 'pajama', 'ear', 'pedestal', 'viaduct',
        'shoji', 'snowplow', 'puck', 'gyromitra', 'birdhouse', 'flatworm',
        'pier', 'coral reef', 'pot', 'mortar', 'polaroid camera',
        'passenger car', 'barracouta', 'banded gecko',
        'black-and-tan coonhound', 'safe', 'ski', 'torch', 'green lizard',
        'volleyball', 'brambling', 'solar dish', 'lawn mower', 'swing',
        'hyena', 'staffordshire bullterrier', 'screw', 'toilet tissue',
        'velvet', 'scale', 'stopwatch', 'sock', 'koala', 'garbage truck',
        'spider monkey', 'afghan hound', 'chain', 'upright', 'flagpole',
        'tree frog', 'cuirass', 'chest', 'groenendael', 'christmas stocking',
        'lakeland terrier', 'perfume', 'neck brace', 'lab coat', 'carbonara',
        'porcupine', 'shower curtain', 'slug', 'pitcher',
        'flat-coated retriever', 'pekinese', 'oscilloscope', 'church', 'lynx',
        'cowboy hat', 'table lamp', 'pug', 'crate', 'water buffalo',
        'labrador retriever', 'weimaraner', 'giant schnauzer', 'stove',
        'sea urchin', 'banjo', 'tiger', 'miniskirt', 'eft',
        'european gallinule', 'vending machine', 'miniature schnauzer',
        'maypole', 'bull mastiff', 'hoopskirt', 'coffeepot', 'four-poster',
        'safety pin', 'monarch', 'beer glass', 'grasshopper', 'head cabbage',
        'parking meter', 'bonnet', 'chiffonier', 'great dane', 'spider web',
        'electric locomotive', 'scotch terrier', 'australian terrier',
        'honeycomb', 'leafhopper', 'beer bottle', 'mud turtle', 'lifeboat',
        'cassette', "potter's wheel", 'oystercatcher', 'space heater',
        'coral fungus', 'sunglass', 'quail', 'triumphal arch', 'collie',
        'walker hound', 'bucket', 'bee', 'komodo dragon', 'dugong', 'gibbon',
        'trailer truck', 'king crab', 'cheetah', 'rifle', 'stingray', 'bison',
        'ipod', 'modem', 'box turtle', 'motor scooter', 'container ship',
        'vestment', 'dingo', 'radiator', 'giant panda', 'nail', 'sea slug',
        'indigo bunting', 'trimaran', 'jacamar', 'chimpanzee', 'comic book',
        'odometer', 'dishwasher', 'bolo tie', 'barn', 'paddlewheel',
        'appenzeller', 'great white shark', 'green snake', 'jackfruit',
        'llama', 'whippet', 'hay', 'leaf beetle', 'sombrero', 'ram',
        'washbasin', 'cup', 'wall clock', 'acorn squash', 'spotted salamander',
        'boston bull', 'border terrier', 'doormat', 'cicada', 'kimono',
        'hand blower', 'ox', 'meerkat', 'space shuttle', 'african hunting dog',
        'violin', 'artichoke', 'toucan', 'bulbul', 'coucal', 'red wolf',
        'seat belt', 'bicycle-built-for-two', 'bow tie', 'pretzel',
        'bedlington terrier', 'albatross', 'punching bag', 'cocktail shaker',
        'diamondback', 'corn', 'ant', 'mountain bike', 'walking stick',
        'standard schnauzer', 'power drill', 'cardigan', 'accordion',
        'wire-haired fox terrier', 'streetcar', 'beach wagon', 'ibizan hound',
        'hair spray', 'car mirror', 'mountain tent', 'trench coat',
        'studio couch', 'pomeranian', 'dough', 'corkscrew', 'broom',
        'parachute', 'band aid', 'water tower', 'teddy', 'fire engine',
        'hornbill', 'hotdog', 'theater curtain', 'crane', 'malinois', 'lion',
        'african elephant', 'handkerchief', 'caldron', 'shopping basket',
        'gown', 'wolf spider', 'vizsla', 'electric ray', 'freight car',
        'pembroke', 'feather boa', 'wallet', 'agama', 'hard disc', 'stretcher',
        'sorrel', 'trilobite', 'basset', 'vulture', 'tarantula', 'hermit crab',
        'king snake', 'robin', 'bernese mountain dog', 'ski mask',
        'fountain pen', 'combination lock', 'yurt', 'clumber', 'park bench',
        'baboon', 'kuvasz', 'centipede', 'tabby', 'steam locomotive', 'badger',
        'irish water spaniel', 'picket fence', 'gong', 'canoe',
        'swimming trunks', 'submarine', 'echidna', 'bib', 'refrigerator',
        'hammer', 'lemon', 'admiral', 'chihuahua', 'basenji', 'pinwheel',
        'golfcart', 'bullet train', 'crib', 'muzzle', 'eggnog',
        'old english sheepdog', 'tray', 'tiger beetle', 'electric guitar',
        'peacock', 'soup bowl', 'wallaby', 'abacus', 'dalmatian', 'harvester',
        'aircraft carrier', 'snowmobile', 'welsh springer spaniel',
        'affenpinscher', 'oboe', 'cassette player', 'pencil sharpener',
        'japanese spaniel', 'plunger', 'black widow', 'norfolk terrier',
        'reflex camera', 'ice bear', 'redbone', 'mongoose', 'warthog',
        'arabian camel', 'bittern', 'mixing bowl', 'tailed frog', 'scabbard',
        'castle', 'curly-coated retriever', 'garden spider', 'folding chair',
        'mouse', 'prayer rug', 'red fox', 'toy terrier', 'leonberg',
        'lycaenid', 'poncho', 'goldfish', 'red-backed sandpiper', 'holster',
        'hair slide', 'coho', 'komondor', 'macaw', 'maltese dog', 'megalith',
        'sarong', 'green mamba', 'sea lion', 'water ouzel', 'bulletproof vest',
        'sulphur-crested cockatoo', 'scottish deerhound', 'steel arch bridge',
        'catamaran', 'brittany spaniel', 'redshank', 'otter',
        'brabancon griffon', 'balloon', 'rule', 'planetarium', 'trombone',
        'mitten', 'abaya', 'crash helmet', 'milk can', 'hartebeest',
        'windsor tie', 'irish terrier', 'african chameleon', 'matchstick',
        'water bottle', 'cloak', 'ground beetle', 'ashcan', 'crane',
        'gila monster', 'unicycle', 'gazelle', 'wombat', 'brain coral',
        'projector', 'custard apple', 'proboscis monkey', 'tibetan mastiff',
        'mosque', 'plastic bag', 'backpack', 'drum', 'norwich terrier',
        'pizza', 'carton', 'plane', 'gorilla', 'jigsaw puzzle', 'forklift',
        'isopod', 'otterhound', 'vacuum', 'european fire salamander', 'apron',
        'langur', 'boxer', 'african grey', 'ice lolly', 'toilet seat',
        'golf ball', 'titi', 'drake', 'ostrich', 'magnetic compass',
        'great pyrenees', 'rhodesian ridgeback', 'buckeye', 'dungeness crab',
        'toy poodle', 'ptarmigan', 'amphibian', 'monitor', 'school bus',
        'schooner', 'spatula', 'weevil', 'speedboat', 'sundial', 'borzoi',
        'bassoon', 'bath towel', 'pill bottle', 'acorn', 'tick', 'briard',
        'thimble', 'brass', 'white wolf', 'boathouse', 'yawl',
        'miniature pinscher', 'barn spider', 'jean', 'water snake', 'dishrag',
        'yorkshire terrier', 'hammerhead', 'typewriter keyboard', 'papillon',
        'ocarina', 'washer', 'standard poodle', 'china cabinet', 'steel drum',
        'swab', 'mobile home', 'german short-haired pointer', 'saluki',
        'bee eater', 'rock python', 'vine snake', 'kelpie', 'harmonica',
        'military uniform', 'reel', 'thatch', 'maraca', 'tricycle',
        'sidewinder', 'parallel bars', 'banana', 'flute', 'paintbrush',
        'sleeping bag', "yellow lady's slipper", 'three-toed sloth',
        'white stork', 'notebook', 'weasel', 'tiger shark', 'football helmet',
        'madagascar cat', 'dowitcher', 'wreck', 'king penguin', 'lighter',
        'timber wolf', 'racket', 'digital watch', 'liner', 'hen',
        'suspension bridge', 'pillow', "carpenter's kit", 'butternut squash',
        'sandal', 'sussex spaniel', 'hip', 'american staffordshire terrier',
        'flamingo', 'analog clock', 'black and gold garden spider',
        'sea cucumber', 'indian elephant', 'syringe', 'lens cap', 'missile',
        'cougar', 'diaper', 'chambered nautilus', 'garter snake',
        'anemone fish', 'organ', 'limousine', 'horse cart', 'jaguar',
        'frilled lizard', 'crutch', 'sea anemone', 'guenon', 'meat loaf',
        'slide rule', 'saltshaker', 'pomegranate', 'acoustic guitar',
        'shopping cart', 'drilling platform', 'nematode', 'chickadee',
        'academic gown', 'candle', 'norwegian elkhound', 'armadillo',
        'horizontal bar', 'orangutan', 'obelisk', 'stone wall', 'cannon',
        'rugby ball', 'ping-pong ball', 'window shade', 'trolleybus',
        'ice cream', 'pop bottle', 'cock', 'harvestman', 'leatherback turtle',
        'killer whale', 'spaghetti squash', 'chain saw', 'stinkhorn',
        'espresso maker', 'loafer', 'bagel', 'ballplayer', 'skunk',
        'chainlink fence', 'earthstar', 'whiptail', 'barrel',
        'kerry blue terrier', 'triceratops', 'chow', 'grey fox', 'sax',
        'binoculars', 'ladybug', 'silky terrier', 'gas pump', 'cradle',
        'whiskey jug', 'french bulldog', 'eskimo dog', 'hog', 'hognose snake',
        'pickup', 'indian cobra', 'hand-held computer', 'printer', 'pole',
        'bald eagle', 'american alligator', 'dumbbell', 'umbrella', 'mink',
        'shower cap', 'tank', 'quill', 'fox squirrel', 'ambulance',
        'lesser panda', 'frying pan', 'letter opener', 'hook', 'strainer',
        'pick', 'dragonfly', 'gar', 'piggy bank', 'envelope', 'stole', 'ibex',
        'american chameleon', 'bearskin', 'microwave', 'petri dish',
        'wood rabbit', 'beacon', 'dung beetle', 'warplane', 'ruddy turnstone',
        'knot', 'fur coat', 'hamper', 'beagle', 'ringlet', 'mask',
        'persian cat', 'cellular telephone', 'american coot', 'apiary',
        'shovel', 'coffee mug', 'sewing machine', 'spoonbill', 'padlock',
        'bell pepper', 'great grey owl', 'squirrel monkey',
        'sulphur butterfly', 'scoreboard', 'bow', 'malamute', 'siamang',
        'snail', 'remote control', 'sea snake', 'loupe', 'model t',
        'english setter', 'dining table', 'face powder', 'tench',
        "jack-o'-lantern", 'croquet ball', 'water jug', 'airedale', 'airliner',
        'guinea pig', 'hare', 'damselfly', 'thresher', 'limpkin', 'buckle',
        'english springer', 'boa constrictor', 'french horn',
        'black-footed ferret', 'shetland sheepdog', 'capuchin', 'cheeseburger',
        'miniature poodle', 'spotlight', 'wooden spoon',
        'west highland white terrier', 'wig', 'running shoe', 'cowboy boot',
        'brown bear', 'iron', 'brassiere', 'magpie', 'gondola', 'grand piano',
        'granny smith', 'mashed potato', 'german shepherd', 'stethoscope',
        'cauliflower', 'soccer ball', 'pay-phone', 'jellyfish', 'cairn',
        'polecat', 'trifle', 'photocopier', 'shih-tzu', 'orange', 'guacamole',
        'hatchet', 'cello', 'egyptian cat', 'basketball', 'moving van',
        'mortarboard', 'dial telephone', 'street sign', 'oil filter', 'beaver',
        'spiny lobster', 'chime', 'bookcase', 'chiton', 'black grouse', 'jay',
        'axolotl', 'oxygen mask', 'cricket', 'worm fence', 'indri',
        'cockroach', 'mushroom', 'dandie dinmont', 'tennis ball',
        'howler monkey', 'rapeseed', 'tibetan terrier', 'newfoundland',
        'dutch oven', 'paddle', 'joystick', 'golden retriever',
        'blenheim spaniel', 'mantis', 'soft-coated wheaten terrier',
        'little blue heron', 'convertible', 'bloodhound', 'palace',
        'medicine chest', 'english foxhound', 'cleaver', 'sweatshirt',
        'mosquito net', 'soap dispenser', 'ladle', 'screwdriver',
        'fire screen', 'binder', 'suit', 'barrow', 'clog', 'cucumber',
        'baseball', 'lorikeet', 'conch', 'quilt', 'eel', 'horned viper',
        'night snake', 'angora', 'pickelhaube', 'gasmask', 'patas')

    # Some too large images are downsampled in LoadImageNetSImageFromFile.
    # These images should be upsampled back in results2img.
    LARGES = {
        '00022800': [1225, 900],
        '00037230': [2082, 2522],
        '00011749': [1000, 1303],
        '00040173': [1280, 960],
        '00027045': [1880, 1330],
        '00019424': [2304, 3072],
        '00015496': [1728, 2304],
        '00025715': [1083, 1624],
        '00008260': [1400, 1400],
        '00047233': [850, 1540],
        '00043667': [2066, 1635],
        '00024274': [1920, 2560],
        '00028437': [1920, 2560],
        '00018910': [1536, 2048],
        '00046074': [1600, 1164],
        '00021215': [1024, 1540],
        '00034174': [960, 1362],
        '00007361': [960, 1280],
        '00030207': [1512, 1016],
        '00015637': [1600, 1200],
        '00013665': [2100, 1500],
        '00028501': [1200, 852],
        '00047237': [1624, 1182],
        '00026950': [1200, 1600],
        '00041704': [1920, 2560],
        '00027074': [1200, 1600],
        '00016473': [1200, 1200],
        '00012206': [2448, 3264],
        '00019622': [960, 1280],
        '00008728': [2806, 750],
        '00027712': [1128, 1700],
        '00007195': [1290, 1824],
        '00002942': [2560, 1920],
        '00037032': [1954, 2613],
        '00018543': [1067, 1600],
        '00041570': [1536, 2048],
        '00004422': [1728, 2304],
        '00044827': [800, 1280],
        '00046674': [1200, 1600],
        '00017711': [1200, 1600],
        '00048488': [1889, 2834],
        '00000706': [1501, 2001],
        '00032736': [1200, 1600],
        '00024348': [1536, 2048],
        '00023430': [1051, 1600],
        '00030496': [1350, 900],
        '00026543': [1280, 960],
        '00010969': [2560, 1920],
        '00025272': [1294, 1559],
        '00019950': [1536, 1024],
        '00004466': [1182, 1722],
        '00029917': [3072, 2304],
        '00014683': [1145, 1600],
        '00013084': [1281, 2301],
        '00039792': [1760, 1034],
        '00046246': [2448, 3264],
        '00004280': [984, 1440],
        '00009435': [1127, 1502],
        '00012860': [1673, 2500],
        '00016702': [1444, 1000],
        '00011278': [2048, 3072],
        '00048174': [1605, 2062],
        '00035451': [1225, 1636],
        '00024769': [1200, 900],
        '00032797': [1251, 1664],
        '00027924': [1453, 1697],
        '00010965': [1536, 2048],
        '00020735': [1200, 1600],
        '00027789': [853, 1280],
        '00015113': [1324, 1999],
        '00037571': [1251, 1586],
        '00030120': [1536, 2048],
        '00044219': [2448, 3264],
        '00024604': [1535, 1955],
        '00010926': [1200, 900],
        '00017509': [1536, 2048],
        '00042373': [924, 1104],
        '00037066': [1536, 2048],
        '00025494': [1880, 1060],
        '00028610': [1377, 2204],
        '00007196': [1202, 1600],
        '00030788': [2592, 1944],
        '00046865': [1920, 2560],
        '00027141': [1600, 1200],
        '00023215': [1200, 1600],
        '00000218': [1439, 1652],
        '00048126': [1516, 927],
        '00030408': [1600, 2400],
        '00038582': [1600, 1200],
        '00046959': [1304, 900],
        '00016988': [1242, 1656],
        '00017201': [1629, 1377],
        '00017658': [1000, 1035],
        '00002766': [1495, 2383],
        '00038573': [1600, 1071],
        '00042297': [1200, 1200],
        '00010564': [995, 1234],
        '00001189': [1600, 1200],
        '00007018': [1858, 2370],
        '00043554': [1200, 1600],
        '00000746': [1200, 1600],
        '00001386': [960, 1280],
        '00029975': [1600, 1200],
        '00016221': [2877, 2089],
        '00003152': [1200, 1600],
        '00002552': [1200, 1600],
        '00009402': [1125, 1500],
        '00040672': [960, 1280],
        '00024540': [960, 1280],
        '00049770': [1457, 1589],
        '00014533': [841, 1261],
        '00006228': [1417, 1063],
        '00034688': [1354, 2032],
        '00032897': [1071, 1600],
        '00024356': [2043, 3066],
        '00019656': [1318, 1984],
        '00035802': [2288, 2001],
        '00017499': [1502, 1162],
        '00046898': [1200, 1600],
        '00040883': [1024, 1280],
        '00031353': [1544, 1188],
        '00028419': [1600, 1200],
        '00048897': [2304, 3072],
        '00040683': [1296, 1728],
        '00042406': [848, 1200],
        '00036007': [900, 1200],
        '00010515': [1688, 1387],
        '00048409': [5005, 3646],
        '00032654': [1200, 1600],
        '00037955': [1200, 1600],
        '00038471': [3072, 2048],
        '00036201': [913, 1328],
        '00038619': [1728, 2304],
        '00038165': [926, 2503],
        '00033240': [1061, 1158],
        '00023086': [1200, 1600],
        '00041385': [1200, 1600],
        '00014066': [2304, 3072],
        '00049973': [1211, 1261],
        '00043188': [2000, 3000],
        '00047186': [1535, 1417],
        '00046975': [1560, 2431],
        '00034402': [1776, 2700],
        '00017033': [1392, 1630],
        '00041068': [1280, 960],
        '00011024': [1317, 900],
        '00048035': [1800, 1200],
        '00033286': [994, 1500],
        '00016613': [1152, 1536],
        '00044160': [888, 1200],
        '00021138': [902, 1128],
        '00022300': [798, 1293],
        '00034300': [1920, 2560],
        '00008603': [1661, 1160],
        '00045173': [2312, 903],
        '00048616': [960, 1280],
        '00048317': [3872, 2592],
        '00045470': [1920, 1800],
        '00043934': [1667, 2500],
        '00010699': [2240, 1488],
        '00030550': [1200, 1600],
        '00010516': [1704, 2272],
        '00001779': [1536, 2048],
        '00018389': [1084, 1433],
        '00013889': [3072, 2304],
        '00022440': [2112, 2816],
        '00024005': [2592, 1944],
        '00046620': [960, 1280],
        '00035227': [960, 1280],
        '00033636': [1110, 1973],
        '00003624': [1165, 1600],
        '00033400': [1200, 1600],
        '00013891': [1200, 1600],
        '00022593': [1472, 1456],
        '00009546': [1936, 2592],
        '00022022': [1182, 1740],
        '00022982': [1200, 1600],
        '00039569': [1600, 1067],
        '00009276': [930, 1240],
        '00026777': [960, 1280],
        '00047680': [1425, 882],
        '00040785': [853, 1280],
        '00002037': [1944, 2592],
        '00005813': [1098, 987],
        '00018328': [1128, 1242],
        '00022318': [1500, 1694],
        '00026654': [790, 1285],
        '00012895': [1600, 1067],
        '00007882': [980, 1024],
        '00043771': [1008, 1043],
        '00032990': [3621, 2539],
        '00034094': [1175, 1600],
        '00034302': [1463, 1134],
        '00025021': [1503, 1520],
        '00000771': [900, 1200],
        '00025149': [1600, 1200],
        '00005211': [1063, 1600],
        '00049544': [1063, 1417],
        '00025378': [1800, 2400],
        '00024287': [1200, 1600],
        '00013550': [2448, 3264],
        '00008076': [1200, 1600],
        '00039536': [1000, 1500],
        '00020331': [1024, 1280],
        '00002623': [1050, 1400],
        '00031071': [873, 1320],
        '00025266': [1024, 1536],
        '00015109': [1213, 1600],
        '00027390': [1200, 1600],
        '00018894': [1584, 901],
        '00049009': [900, 1203],
        '00026671': [1201, 1601],
        '00018668': [1024, 990],
        '00016942': [1024, 1024],
        '00046430': [1944, 3456],
        '00033261': [1341, 1644],
        '00017363': [2304, 2898],
        '00045935': [2112, 2816],
        '00027084': [900, 1200],
        '00037716': [1611, 981],
        '00030879': [1200, 1600],
        '00027539': [1534, 1024],
        '00030052': [1280, 852],
        '00011015': [2808, 2060],
        '00037004': [1920, 2560],
        '00044012': [2240, 1680],
        '00049818': [1704, 2272],
        '00003541': [1200, 1600],
        '00000520': [2448, 3264],
        '00028331': [3264, 2448],
        '00030244': [1200, 1600],
        '00039079': [1600, 1200],
        '00033432': [1600, 1200],
        '00010533': [1200, 1600],
        '00005916': [899, 1200],
        '00038903': [1052, 1592],
        '00025169': [1895, 850],
        '00049042': [1200, 1600],
        '00021828': [1280, 988],
        '00013420': [3648, 2736],
        '00045201': [1381, 1440],
        '00021857': [776, 1296],
        '00048810': [1168, 1263],
        '00047860': [2592, 3888],
        '00046960': [2304, 3072],
        '00039357': [1200, 1600],
        '00019620': [1536, 2048],
        '00026710': [1944, 2592],
        '00021277': [1079, 1151],
        '00028387': [1128, 1585],
        '00028796': [990, 1320],
        '00035149': [1064, 1600],
        '00020182': [1843, 1707],
        '00018286': [2592, 1944],
        '00035658': [1488, 1984],
        '00008180': [1024, 1633],
        '00018740': [1200, 1600],
        '00044356': [1536, 2048],
        '00038857': [1252, 1676],
        '00035014': [1200, 1600],
        '00044824': [1200, 1600],
        '00009912': [1200, 1600],
        '00014572': [2400, 1800],
        '00001585': [1600, 1067],
        '00047704': [1200, 1600],
        '00038537': [920, 1200],
        '00027941': [2200, 3000],
        '00028526': [2592, 1944],
        '00042353': [1280, 1024],
        '00043409': [2000, 1500],
        '00002209': [2592, 1944],
        '00040841': [1613, 1974],
        '00038889': [900, 1200],
        '00046941': [1200, 1600],
        '00014029': [846, 1269],
        '00023091': [900, 1200],
        '00036184': [877, 1350],
        '00006165': [1200, 1600],
        '00033991': [868, 2034],
        '00035078': [1680, 2240],
        '00045681': [1467, 1134],
        '00043867': [1200, 1600],
        '00003586': [1200, 1600],
        '00039024': [1283, 2400],
        '00048990': [1200, 1200],
        '00044334': [960, 1280],
        '00020939': [960, 1280],
        '00031529': [1302, 1590],
        '00014867': [2112, 2816],
        '00034239': [1536, 2048],
        '00031845': [1200, 1600],
        '00045721': [1536, 2048],
        '00025336': [1441, 1931],
        '00040323': [900, 1152],
        '00009133': [876, 1247],
        '00033687': [2357, 3657],
        '00038351': [1306, 1200],
        '00022618': [1060, 1192],
        '00001626': [777, 1329],
        '00039137': [1071, 1600],
        '00034896': [1426, 1590],
        '00048502': [1187, 1837],
        '00048077': [1712, 2288],
        '00026239': [1200, 1600],
        '00032687': [857, 1280],
        '00006639': [1498, 780],
        '00037738': [2112, 2816],
        '00035760': [1123, 1447],
        '00004897': [1083, 1393],
        '00012141': [3584, 2016],
        '00016278': [3234, 2281],
        '00006661': [1787, 3276],
        '00033040': [1200, 1800],
        '00009881': [960, 1280],
        '00008240': [2592, 1944],
        '00023506': [960, 1280],
        '00046982': [1693, 2480],
        '00049632': [2310, 1638],
        '00005473': [960, 1280],
        '00013491': [2000, 3008],
        '00005581': [1593, 1200],
        '00005196': [1417, 2133],
        '00049433': [1207, 1600],
        '00012323': [1200, 1800],
        '00021883': [1600, 2400],
        '00031877': [2448, 3264],
        '00046428': [1200, 1600],
        '00000725': [881, 1463],
        '00044936': [894, 1344],
        '00012054': [3040, 4048],
        '00025447': [900, 1200],
        '00005290': [1520, 2272],
        '00023326': [984, 1312],
        '00047891': [1067, 1600],
        '00026115': [1067, 1600],
        '00010051': [1062, 1275],
        '00005999': [1123, 1600],
        '00021752': [1071, 1600],
        '00041559': [1200, 1600],
        '00025931': [836, 1410],
        '00009327': [2848, 4288],
        '00029735': [1905, 1373],
        '00012922': [1024, 1547],
        '00042259': [1548, 1024],
        '00024949': [1050, 956],
        '00014669': [900, 1200],
        '00028028': [1170, 1730],
        '00003183': [1152, 1535],
        '00039304': [1050, 1680],
        '00014939': [1904, 1240],
        '00048366': [1600, 1200],
        '00022406': [3264, 2448],
        '00033363': [1125, 1500],
        '00041230': [1125, 1500],
        '00044222': [2105, 2472],
        '00021950': [1200, 1200],
        '00028475': [2691, 3515],
        '00002149': [900, 1600],
        '00033356': [1080, 1920],
        '00041158': [960, 1280],
        '00029672': [1536, 2048],
        '00045816': [1023, 1153],
        '00020471': [2076, 2716],
        '00012398': [1067, 1600],
        '00017884': [2048, 3072],
        '00025132': [1200, 1600],
        '00042429': [1362, 1980],
        '00021285': [1127, 1200],
        '00045113': [2792, 2528],
        '00047915': [1200, 891],
        '00009481': [1097, 924],
        '00025448': [1760, 2400],
        '00033911': [1759, 2197],
        '00044684': [1200, 1600],
        '00033754': [2304, 1728],
        '00002733': [1536, 2048],
        '00027371': [936, 1128],
        '00019941': [685, 1591],
        '00028479': [1944, 2592],
        '00018451': [1028, 1028],
        '00024067': [1000, 1352],
        '00016524': [1704, 2272],
        '00048926': [1944, 2592],
        '00020992': [1024, 1280],
        '00044576': [1024, 1280],
        '00031796': [960, 1280],
        '00043540': [2448, 3264],
        '00049250': [1056, 1408],
        '00030602': [2592, 3872],
        '00046571': [1118, 1336],
        '00024908': [1442, 1012],
        '00018903': [3072, 2304],
        '00032370': [1944, 2592],
        '00043445': [1050, 1680],
        '00030791': [2228, 3168],
        '00046866': [2057, 3072],
        '00047293': [1800, 2400],
        '00024853': [1296, 1936],
        '00014344': [1125, 1500],
        '00041327': [960, 1280],
        '00017867': [2592, 3872],
        '00037615': [1664, 2496],
        '00011247': [1605, 2934],
        '00034664': [2304, 1728],
        '00013733': [1024, 1280],
        '00009125': [1200, 1600],
        '00035163': [1654, 1233],
        '00017537': [1200, 1600],
        '00043423': [1536, 2048],
        '00035755': [1154, 900],
        '00021712': [1600, 1200],
        '00000597': [2792, 1908],
        '00033579': [882, 1181],
        '00035830': [2112, 2816],
        '00005917': [920, 1380],
        '00029722': [2736, 3648],
        '00039979': [1200, 1600],
        '00040854': [1606, 2400],
        '00039884': [2848, 4288],
        '00003508': [1128, 1488],
        '00019862': [1200, 1600],
        '00041813': [1226, 1160],
        '00007121': [985, 1072],
        '00013315': [883, 1199],
        '00049822': [922, 1382],
        '00027622': [1434, 1680],
        '00047689': [1536, 2048],
        '00017415': [1491, 2283],
        '00023713': [927, 1287],
        '00001632': [1200, 1600],
        '00033104': [1200, 1600],
        '00017643': [1002, 1200],
        '00038396': [1330, 1999],
        '00027614': [2166, 2048],
        '00025962': [1600, 1200],
        '00015915': [1067, 1600],
        '00008940': [1942, 2744],
        '00012468': [2000, 2000],
        '00046953': [828, 1442],
        '00002084': [1067, 1600],
        '00040245': [2657, 1898],
        '00023718': [900, 1440],
        '00022770': [924, 1280],
        '00028957': [960, 1280],
        '00001054': [2048, 3072],
        '00040541': [1369, 1809],
        '00024869': [960, 1280],
        '00037655': [900, 1440],
        '00037200': [2171, 2575],
        '00037390': [1394, 1237],
        '00025318': [1054, 1024],
        '00021634': [1800, 2400],
        '00044217': [1003, 1024],
        '00014877': [1200, 1600],
        '00029504': [1224, 1632],
        '00016422': [960, 1280],
        '00028015': [1944, 2592],
        '00006235': [967, 1291],
        '00045909': [2272, 1704]
    }

    def __init__(self, subset=919, **kwargs):

        assert subset in (50, 300, 919), \
            'ImageNet-S has three subsets, i.e., '\
            'ImageNet-S50, ImageNet-S300 and ImageNet-S919.'
        if subset == 50:
            self.CLASSES = self.CLASSES50
        elif subset == 300:
            self.CLASSES = self.CLASSES300
        else:
            self.CLASSES = self.CLASSES919

        super(ImageNetSDataset, self).__init__(
            img_suffix='.JPEG',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            ignore_index=1000,
            **kwargs)

        self.subset = subset
        gt_seg_map_loader_cfg = kwargs.get('gt_seg_map_loader_cfg', None)
        self.gt_seg_map_loader = LoadImageNetSAnnotations(
        ) if gt_seg_map_loader_cfg is None else LoadImageNetSAnnotations(
            **gt_seg_map_loader_cfg)

    def pre_eval(self, preds, indices):
        """Collect eval result for ImageNet-S. In LoadImageNetSImageFromFile,
        the too large images have been downsampled. Here the preds should be
        upsampled back after argmax.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            seg_map = self.get_gt_seg_map_by_idx(index)
            pred = mmcv.imresize(
                pred,
                size=(seg_map.shape[1], seg_map.shape[0]),
                interpolation='nearest')
            pre_eval_results.append(
                intersect_and_union(
                    pred,
                    seg_map,
                    len(self.CLASSES),
                    self.ignore_index,
                    # as the labels has been converted when dataset initialized
                    # in `get_palette_for_custom_classes ` this `label_map`
                    # should be `dict()`, see
                    # https://github.com/open-mmlab/mmsegmentation/issues/1415
                    # for more ditails
                    label_map=dict(),
                    reduce_zero_label=self.reduce_zero_label))

        return pre_eval_results

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images for ImageNetS. The results
        should be converted as RGB images due to 919 (>256) categroies. In
        LoadImageNetSImageFromFile, the too large images have been downsampled.
        Here the results should be upsampled back after argmax.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']

            directory = filename.split('/')[-2]
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, directory,
                                    f'{basename}.png')

            # The index range of output is from 0 to 919/300/50.
            result_rgb = np.zeros(shape=(result.shape[0], result.shape[1], 3))
            result_rgb[:, :, 0] = result % 256
            result_rgb[:, :, 1] = result // 256

            if basename.split('_')[2] in self.LARGES.keys():
                result_rgb = mmcv.imresize(
                    result_rgb,
                    size=(self.LARGES[basename.split('_')[2]][1],
                          self.LARGES[basename.split('_')[2]][0]),
                    interpolation='nearest')

            mmcv.mkdir_or_exist(osp.join(imgfile_prefix, directory))
            output = Image.fromarray(result_rgb.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for ImageNetS
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)

        return result_files
