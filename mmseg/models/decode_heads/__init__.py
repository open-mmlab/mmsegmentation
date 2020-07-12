from .ann_head import ANNHead
from .aspp_head import ASPPHead
from .cc_head import CCHead
from .da_head import DAHead
from .enc_head import EncHead
from .fcn_head import FCNHead
from .gc_head import GCHead
from .nl_head import NLHead
from .ocr_head import OCRHead
from .ocr_head_v2 import OCRHeadv2
from .ocr_head_v3 import OCRHeadv3
from .ocrplus_head import OCRPlusHead
from .sep_ocr_head import DepthwiseSeparableOCRHead, DepthwiseSeparableOCRHeadv2, DepthwiseSeparableOCRHeadv3, DepthwiseSeparableOCRHeadv4
from .sep_ocrplus_head import DepthwiseSeparableOCRPlusHead, DepthwiseSeparableOCRPlusHeadv2
from .psa_head import PSAHead
from .psp_head import PSPHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .uper_head import UPerHead

__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'PSAHead', 'NLHead', 'GCHead', 'CCHead',
    'UPerHead', 'DepthwiseSeparableASPPHead', 'ANNHead', 'DAHead', 'OCRHead',
    'EncHead', 'OCRHeadv2', 'OCRHeadv3', 'OCRPlusHead', 'DepthwiseSeparableOCRPlusHead', 'DepthwiseSeparableOCRPlusHeadv2',
    'DepthwiseSeparableOCRHead', 'DepthwiseSeparableOCRHeadv2', 'DepthwiseSeparableOCRHeadv3', 'DepthwiseSeparableOCRHeadv4'
]
