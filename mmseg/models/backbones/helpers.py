import torch
import torch.nn.functional as F
import math
import logging
import warnings
import errno
import os
import sys
import re
import zipfile
from urllib.parse import urlparse  # noqa: F401

HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
_logger = logging.getLogger(__name__)


def load_state_dict_from_url(url, model_dir=None, file_name=None, check_hash=False, progress=True, map_location=None):
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn(
            'TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    if model_dir is None:
        hub_dir = torch.hub.get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')
        try:
            os.makedirs(model_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                # Directory already exists, ignore.
                pass
            else:
                # Unexpected OSError, re-raise.
                raise
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = HASH_REGEX.search(
            filename).group(1) if check_hash else None
        torch.hub.download_url_to_file(
            url, cached_file, hash_prefix, progress=progress)
    if zipfile.is_zipfile(cached_file):
        state_dict = torch.load(
            cached_file, map_location=map_location)['model']
    else:
        state_dict = torch.load(cached_file, map_location=map_location)
    return state_dict


def load_pretrained(model, cfg=None, num_classes=1000, in_chans=3, filter_fn=None, strict=True, pos_embed_interp=False, num_patches=576, align_corners=False):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning(
            "Pretrained model URL is invalid, using random initialization.")
        return

    if 'pretrained_finetune' in cfg and cfg['pretrained_finetune']:
        state_dict = torch.load(cfg['pretrained_finetune'])
        print('load pre-trained weight from ' + cfg['pretrained_finetune'])
    else:
        state_dict = load_state_dict_from_url(
            cfg['url'], progress=False, map_location='cpu')
        print('load pre-trained weight from imagenet21k')

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        _logger.info(
            'Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        # Some weights are in torch.half, ensure it's float for sum on CPU
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight
    elif in_chans != 3:
        conv1_name = cfg['first_conv']
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I == 3:
            _logger.warning(
                'Deleting first conv (%s) from pretrained weights.' % conv1_name)
            del state_dict[conv1_name + '.weight']
            strict = False
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            _logger.info(
                'Repeating first conv (%s) weights in channel dim.' % conv1_name)
            repeat = int(math.ceil(in_chans / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[
                :, :in_chans, :, :]
            conv1_weight *= (3 / float(in_chans))
            conv1_weight = conv1_weight.to(conv1_type)
            state_dict[conv1_name + '.weight'] = conv1_weight

    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != cfg['num_classes']:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    if pos_embed_interp:
        n, c, hw = state_dict['pos_embed'].transpose(1, 2).shape
        h = w = int(math.sqrt(hw))
        pos_embed_weight = state_dict['pos_embed'][:, (-h * w):]
        pos_embed_weight = pos_embed_weight.transpose(1, 2)
        n, c, hw = pos_embed_weight.shape
        h = w = int(math.sqrt(hw))
        pos_embed_weight = pos_embed_weight.view(n, c, h, w)

        pos_embed_weight = F.interpolate(pos_embed_weight, size=int(
            math.sqrt(num_patches)), mode='bilinear', align_corners=align_corners)
        pos_embed_weight = pos_embed_weight.view(n, c, -1).transpose(1, 2)

        cls_token_weight = state_dict['pos_embed'][:, 0].unsqueeze(1)

        state_dict['pos_embed'] = torch.cat(
            (cls_token_weight, pos_embed_weight), dim=1)

    model.load_state_dict(state_dict, strict=strict)
