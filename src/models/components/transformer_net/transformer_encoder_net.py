import math

from torch import nn
import torch
from typing import Optional, Dict, Union, Tuple

from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torchinfo import summary

from src.models.components.transformer_net.classification_head import ClassificationHead
from src.models.components.transformer_net.patch_embedding import PatchEmbedding, PatchInputMode
from src.models.components.transformer_net.positional_embedding import PositionalEncoding
from src.models.components.transformer_net.special_embedding_type import SpecialEmbeddingType
from src.models.components.transformer_net.transformer_encoder_net_input_type import TransformerEncoderNetInputType


class TransformerEncoderNet(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    # pytorch code: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    # Constructor
    def __init__(
            self,
            dim_model: int = 768,
            dim_feedforward: int = 2048,
            out_size: int = 768,
            num_heads: int = 8,
            num_encoder_layers: int = 12,
            dropout_p: float = 0.1,
            special_embedding_type: Optional[SpecialEmbeddingType] = None,
            use_positional_embeddings: bool = False,
            use_mean_of_outputs_instead_of_cls: bool = False,
            input_type: TransformerEncoderNetInputType = TransformerEncoderNetInputType.DIRECT,
            patch_embedding_layer_configs: Optional[Dict] = None
    ):
        super().__init__()
        self.use_mean_of_outputs_instead_of_cls = use_mean_of_outputs_instead_of_cls
        self.use_positional_embeddings = use_positional_embeddings
        self.model_type = "Transformer_Encoder"
        self.dim_model = dim_model
        self.positional_encoder = None
        if use_positional_embeddings:
            self.positional_encoder = PositionalEncoding(
                dim_model=dim_model, dropout_p=dropout_p, max_len=5000
            )
        self.special_embeddings = None
        if isinstance(input_type, str):
            input_type = TransformerEncoderNetInputType(input_type)
        self.input_type = input_type
        if isinstance(special_embedding_type, str):
            special_embedding_type = SpecialEmbeddingType(special_embedding_type)
        self.special_embedding_type = special_embedding_type
        if special_embedding_type is SpecialEmbeddingType.CLS_SEP:
            self.special_embeddings = nn.Embedding(2, dim_model)
        if special_embedding_type is SpecialEmbeddingType.CLS_SEP_VIS_TEXT:
            self.special_embeddings = nn.Embedding(4, dim_model)

        encoder_layers = TransformerEncoderLayer(d_model=dim_model,
                                                 nhead=num_heads,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout_p)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      num_layers=num_encoder_layers)
        self.classification_head = ClassificationHead(emb_size=dim_model,
                                                      out_size=out_size,
                                                      use_mean_of_outputs_instead_of_cls=use_mean_of_outputs_instead_of_cls)
        self.patch_embedding_layer_configs = patch_embedding_layer_configs
        if self.patch_embedding_layer_configs is not None \
                and self.input_type == TransformerEncoderNetInputType.PATCHED_IMAGE_EMBEDDING:
            self.patch_embedding = PatchEmbedding(**self.patch_embedding_layer_configs)

    def forward(self,
                src: Union[torch.Tensor, Tuple[torch.Tensor,
                                               torch.Tensor,
                                               Optional[torch.Tensor]]]
                ) -> torch.Tensor:
        """
        Args:
            src ():   (seq, batch, feature) ||
                (    # intra_panel_vision_context: panel_count, batch, embedding_dim
                 intra_panel_text_context: panel_count, intra_panel_text_seq_len, batch, embedding dim),
                 textbox_attn_mask: panel_count, textbox_count, batch (1 - 0)
                 padding source: https://stackoverflow.com/questions/62170439/difference-between-src-mask-and-src-key-padding-mask/68396781#68396781
                src_key_padding_mask: (S) for unbatched input otherwise (N, S)
        Returns:
            torch.Tensor
        """
        if self.input_type == TransformerEncoderNetInputType.PATCHED_IMAGE_EMBEDDING:
            assert not torch.is_tensor(
                src), 'PATCHED IMAGE transformer encoder can not accept tensor src, input should be in tuple...'
            intra_panel_vision_context = src[0]  # panel_count, batch, embedding_dim
            intra_panel_vision_context = self.patch_embedding(
                intra_panel_vision_context)  # panel_count, patch_count, batch, patch_embedding_dim
            intra_panel_text_context = src[1]
            intra_panel_textbox_attn_mask = src[2] if len(src) == 3 else None
            src = (intra_panel_vision_context, intra_panel_text_context, intra_panel_textbox_attn_mask)
        padding_mask = None
        if torch.is_tensor(src):
            if self.special_embedding_type is SpecialEmbeddingType.CLS_SEP:
                # TODO: @gsoykan impl mask_padding feature here too!!
                src = self._basic_add_cls_sep_embeddings_to_src(src)
        else:
            if self.special_embedding_type is SpecialEmbeddingType.CLS_SEP:
                src, padding_mask = self._add_cls_sep_embeddings_to_src(src)
            elif self.special_embedding_type is SpecialEmbeddingType.CLS_SEP_VIS_TEXT:
                # This implies that both text and image data is coming
                src, padding_mask = self._add_cls_sep_vis_text_embeddings_to_src(src)
        src = src * math.sqrt(self.dim_model)
        if self.use_positional_embeddings:
            src = self.positional_encoder(src)
        out = self.transformer_encoder.forward(src, src_key_padding_mask=padding_mask)  # (seq, batch, feature)
        out = self.classification_head.forward(out)  # (batch, out_size)
        return out

    def _basic_add_cls_sep_embeddings_to_src(self, src: torch.Tensor) -> torch.Tensor:
        cls_tokens = torch.full((1, src.shape[1]), 0, dtype=torch.long, device=src.device)
        sep_tokens = torch.full((1, src.shape[1]), 1, dtype=torch.long, device=src.device)
        cls_embeddings = self.special_embeddings(cls_tokens)
        sep_embeddings = self.special_embeddings(sep_tokens)
        return torch.cat([cls_embeddings, src, sep_embeddings], dim=0)

    def _add_cls_sep_embeddings_to_src(self, src: Tuple[Optional[torch.Tensor],
                                                        Optional[torch.Tensor],
                                                        Optional[torch.Tensor]]) -> Tuple[torch.Tensor,
                                                                                          Optional[torch.Tensor]]:
        intra_panel_vision_context = src[0]
        intra_panel_text_context = src[1]
        intra_panel_textbox_attn_mask = src[2] if len(src) == 3 else None
        if intra_panel_vision_context is None and intra_panel_text_context is not None:
            # Means that we are dealing with text only model
            return self._add_cls_sep_embeddings_to_text_src((intra_panel_text_context, intra_panel_textbox_attn_mask))
        elif intra_panel_vision_context is not None and intra_panel_text_context is None:
            # Means that we are dealing with vision only model
            if self.input_type == TransformerEncoderNetInputType.PATCHED_IMAGE_EMBEDDING:
                src = self._add_cls_sep_embeddings_to_patched_vision_src(intra_panel_vision_context)
            else:
                src = self._basic_add_cls_sep_embeddings_to_src(intra_panel_vision_context)
            return src, None
        elif intra_panel_vision_context is not None and intra_panel_text_context is not None:
            # Means that we are dealing with both text and image data
            raise Exception('cls - sep embedding input for text + image data has not been implemented yet!')
        else:
            raise Exception('Unhandled cls - sep embedding input!')

    def _add_cls_sep_embeddings_to_patched_vision_src(self,
                                                      patched_intra_panel_vision_context: torch.Tensor) -> torch.Tensor:
        device = patched_intra_panel_vision_context.device
        panel_count = patched_intra_panel_vision_context.shape[0]
        single_panel_patch_count = patched_intra_panel_vision_context.shape[1]
        batch_size = patched_intra_panel_vision_context.shape[2]
        cls_tokens = torch.full((1, batch_size), 0, dtype=torch.long, device=device)
        sep_tokens = torch.full((panel_count, batch_size), 1, dtype=torch.long, device=device)
        all_tokens = torch.cat((cls_tokens, sep_tokens), dim=0)
        all_embeddings = self.special_embeddings(all_tokens)
        input_tensors = [all_embeddings[0]]
        for i in range(panel_count):
            for j in range(single_panel_patch_count):
                input_tensors.append(patched_intra_panel_vision_context[i][j])
            sep_token_index = 1 + i
            input_tensors.append(all_embeddings[sep_token_index])
        src = torch.stack(input_tensors, dim=0)  # (S, B, E)
        return src

    def _add_cls_sep_embeddings_to_text_src(self,
                                            src: Tuple[torch.Tensor, Optional[torch.Tensor]]) -> Tuple[torch.Tensor,
                                                                                                       torch.Tensor]:
        intra_panel_text_context = src[0]
        intra_panel_textbox_attn_mask = src[1]
        device = intra_panel_text_context.device
        panel_count = intra_panel_text_context.shape[0]
        panel_text_box_count = intra_panel_text_context.shape[1]
        batch_size = intra_panel_text_context.shape[2]
        cls_tokens = torch.full((1, batch_size), 0, dtype=torch.long, device=device)
        sep_tokens = torch.full((panel_count, batch_size), 1, dtype=torch.long, device=device)
        all_tokens = torch.cat((cls_tokens, sep_tokens), dim=0)
        all_embeddings = self.special_embeddings(all_tokens)
        input_tensors = [all_embeddings[0]]
        padding_mask = [torch.zeros((batch_size), device=device).bool()]
        for i in range(panel_count):
            for j in range(panel_text_box_count):
                input_tensors.append(intra_panel_text_context[i][j])
                if intra_panel_textbox_attn_mask is not None:
                    padding_mask.append(intra_panel_textbox_attn_mask[i][j] == 0)
                else:
                    padding_mask.append(torch.zeros((batch_size), device=device).bool())
            sep_token_index = 1 + i
            input_tensors.append(all_embeddings[sep_token_index])
            padding_mask.append(torch.zeros((batch_size), device=device).bool())
        src = torch.stack(input_tensors, dim=0)  # (S, B, E)
        padding_mask = torch.stack(padding_mask).transpose(0, 1)  # (B, S)
        return src, padding_mask

    def _add_cls_sep_vis_text_embeddings_to_src(self, src: Tuple[torch.Tensor,
                                                                 torch.Tensor,
                                                                 Optional[torch.Tensor]]) -> Tuple[torch.Tensor,
                                                                                                   torch.Tensor]:
        intra_panel_vision_context = src[0]
        is_image_patched = True if len(intra_panel_vision_context.shape) == 4 else False
        image_patch_count = 1
        if is_image_patched:
            image_patch_count = intra_panel_vision_context.shape[1]
        intra_panel_text_context = src[1]
        intra_panel_textbox_attn_mask = src[2] if len(src) == 3 else None
        device = intra_panel_text_context.device
        panel_count = intra_panel_text_context.shape[0]
        panel_text_box_count = intra_panel_text_context.shape[1]
        batch_size = intra_panel_text_context.shape[2]
        cls_tokens = torch.full((1, batch_size), 0, dtype=torch.long, device=device)
        sep_tokens = torch.full((panel_count, batch_size), 1, dtype=torch.long, device=device)
        text_tokens = torch.full((panel_text_box_count * panel_count, batch_size), 2, dtype=torch.long, device=device)
        vis_tokens = torch.full((panel_count, batch_size), 3, dtype=torch.long, device=device)
        all_tokens = torch.cat((cls_tokens, sep_tokens, text_tokens, vis_tokens), dim=0)
        all_embeddings = self.special_embeddings(all_tokens)
        input_tensors = [all_embeddings[0]]
        padding_mask = [torch.zeros((batch_size), device=device).bool()]
        for i in range(panel_count):
            vis_embedding_index = 1 + panel_count + panel_text_box_count * panel_count + i
            # print(f'vis embedding index {str(vis_embedding_index)}')
            input_tensors.append(all_embeddings[vis_embedding_index])
            padding_mask.append(torch.zeros((batch_size), device=device).bool())
            if is_image_patched:
                for p in range(image_patch_count):
                    input_tensors.append(intra_panel_vision_context[i][p])
                    padding_mask.append(torch.zeros((batch_size), device=device).bool())
            else:
                input_tensors.append(intra_panel_vision_context[i])
                padding_mask.append(torch.zeros((batch_size), device=device).bool())
            for j in range(panel_text_box_count):
                text_embedding_index = 1 + panel_count + panel_text_box_count * i + j
                # print(f'text embedding index {str(text_embedding_index)}')
                input_tensors.append(all_embeddings[text_embedding_index])
                input_tensors.append(intra_panel_text_context[i][j])
                if intra_panel_textbox_attn_mask is not None:
                    padding_mask.append(intra_panel_textbox_attn_mask[i][j] == 0)
                    padding_mask.append(intra_panel_textbox_attn_mask[i][j] == 0)
                else:
                    padding_mask.append(torch.zeros((batch_size), device=device).bool())
                    padding_mask.append(torch.zeros((batch_size), device=device).bool())
            sep_token_index = 1 + i
            input_tensors.append(all_embeddings[sep_token_index])
            padding_mask.append(torch.zeros((batch_size), device=device).bool())
        src = torch.stack(input_tensors, dim=0)  # (S, B, E)
        padding_mask = torch.stack(padding_mask).transpose(0, 1)  # (B, S)
        return src, padding_mask


# TODO: @gsoykan turn those into tests
####################################################################################################

# intra_panel_vision_context: panel_count, batch, embedding_dim
# intra_panel_text_context: panel_count, intra_panel_text_seq_len, batch, embedding dim)
def check_cls_sep_vis_text_token_insertion():
    dim_model = 256
    panel_count = 10
    batch_size = 8
    intra_panel_text_seq_len = 3
    encoder_net = TransformerEncoderNet(dim_model=dim_model,
                                        use_positional_embeddings=True,
                                        special_embedding_type=SpecialEmbeddingType.CLS_SEP_VIS_TEXT,
                                        use_mean_of_outputs_instead_of_cls=True,
                                        out_size=256,
                                        dim_feedforward=1024,
                                        num_heads=8,
                                        num_encoder_layers=6)
    src = (torch.rand((panel_count, batch_size, dim_model)),
           torch.rand((panel_count, intra_panel_text_seq_len, batch_size, dim_model)),
           torch.ones((panel_count, intra_panel_text_seq_len, batch_size)))
    updated, padding_mask = encoder_net._add_cls_sep_vis_text_embeddings_to_src(src)
    print(updated.shape)


def check_cls_sep_vis_text_token_insertion_image_embedding_patched():
    dim_model = 256
    panel_count = 10
    batch_size = 8
    intra_panel_text_seq_len = 3
    image_patch_count = 4
    encoder_net = TransformerEncoderNet(dim_model=dim_model,
                                        use_positional_embeddings=True,
                                        special_embedding_type=SpecialEmbeddingType.CLS_SEP_VIS_TEXT,
                                        use_mean_of_outputs_instead_of_cls=True,
                                        out_size=256,
                                        dim_feedforward=1024,
                                        num_heads=8,
                                        num_encoder_layers=6,
                                        input_type=TransformerEncoderNetInputType.PATCHED_IMAGE_EMBEDDING,
                                        patch_embedding_layer_configs={'input_mode': PatchInputMode.IMAGE_EMBEDDING,
                                                                       'patch_size': dim_model})
    src = (torch.rand((panel_count, image_patch_count, batch_size, dim_model)),
           torch.rand((panel_count, intra_panel_text_seq_len, batch_size, dim_model)),
           torch.ones((panel_count, intra_panel_text_seq_len, batch_size)))
    updated, padding_mask = encoder_net._add_cls_sep_vis_text_embeddings_to_src(src)
    print(updated.shape)


def check_cls_sep_forward_for_image_data():
    dim_model = 256
    panel_count = 10
    batch_size = 8
    intra_panel_text_seq_len = 3
    encoder_net = TransformerEncoderNet(dim_model=dim_model,
                                        use_positional_embeddings=True,
                                        special_embedding_type=SpecialEmbeddingType.CLS_SEP,
                                        use_mean_of_outputs_instead_of_cls=True,
                                        out_size=256,
                                        dim_feedforward=256,
                                        num_heads=2,
                                        num_encoder_layers=4)
    src = (torch.rand((panel_count, batch_size, dim_model)),
           None,
           None)
    res = encoder_net(src)
    print(res.shape)


def check_cls_sep_forward_for_text_data():
    dim_model = 256
    panel_count = 10
    batch_size = 8
    intra_panel_text_seq_len = 3
    encoder_net = TransformerEncoderNet(dim_model=dim_model,
                                        use_positional_embeddings=True,
                                        special_embedding_type=SpecialEmbeddingType.CLS_SEP,
                                        use_mean_of_outputs_instead_of_cls=True,
                                        out_size=256,
                                        dim_feedforward=256,
                                        num_heads=2,
                                        num_encoder_layers=4)
    src = (None,
           torch.rand((panel_count, intra_panel_text_seq_len, batch_size, dim_model)),
           torch.ones((panel_count, intra_panel_text_seq_len, batch_size)))
    res = encoder_net(src)
    print(res.shape)


def check_model_run():
    dim_model = 768
    encoder_net = TransformerEncoderNet(dim_model=dim_model,
                                        use_positional_embeddings=True,
                                        special_embedding_type=SpecialEmbeddingType.CLS_SEP,
                                        use_mean_of_outputs_instead_of_cls=True)
    src = torch.rand((10, 32, dim_model))
    output = encoder_net(src)
    print(output)
    summary(encoder_net, input_size=(10, 32, dim_model))


if __name__ == '__main__':
    # check_cls_sep_forward_for_image_data()
    # check_cls_sep_forward_for_text_data()
    # check_cls_sep_vis_text_token_insertion()
    # check_model_run()
    check_cls_sep_vis_text_token_insertion_image_embedding_patched()
