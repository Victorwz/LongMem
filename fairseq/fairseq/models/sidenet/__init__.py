# Copyright (c) Facebook Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from fairseq.models.sidenet.transformer_sidenet_config import (
    TransformerSideNetConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
# from fairseq.models.transformer.transformer_decoder import TransformerDecoder, TransformerDecoderBase, Linear
# from fairseq.models.transformer.transformer_encoder import TransformerEncoder, TransformerEncoderBase
# from fairseq.models.transformer.transformer_legacy import (
#     TransformerModel,
#     base_architecture,
#     tiny_architecture,
#     transformer_iwslt_de_en,
#     transformer_wmt_en_de,
#     transformer_vaswani_wmt_en_de_big,
#     transformer_vaswani_wmt_en_fr_big,
#     transformer_wmt_en_de_big,
#     transformer_wmt_en_de_big_t2t,
# )
# from fairseq.models.transformer.transformer_base import TransformerModelBase, Embedding
from .transformer_decoder_sidenet import TransformerDecoderSideNet
from .transformer_decoder_sidenet_bloom import TransformerDecoderSideNetBloom

__all__ = [
    "TransformerSideNetConfig",
    "TransformerDecoderSideNet",
    "TransformerDecoderSideNetBloom",
]
