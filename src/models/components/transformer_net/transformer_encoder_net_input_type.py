from enum import Enum


class TransformerEncoderNetInputType(str, Enum):
    DIRECT = 'DIRECT'
    PATCHED_IMAGE_EMBEDDING = 'PATCHED_IMAGE_EMBEDDING'
