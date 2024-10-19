from enum import Enum


class SpecialEmbeddingType(str, Enum):
    CLS_SEP = 'CLS_SEP'
    # TODO: @gsoykan implement and use masking of unused textbox indices
    CLS_SEP_VIS_TEXT = 'CLS_SEP_VIS_TEXT'
