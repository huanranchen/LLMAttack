import torch
from torch import nn, Tensor
from torch.nn import functional as F
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel
import open_clip
from open_clip import CLIP
from open_clip.transformer import text_global_pool
from .BaseModel import BaseModel
from typing import Optional, List


__all__ = ["TransformerClipAttackLanguageModel", "OpenClipAttackLanguageModel"]


class TransformerClipAttackLanguageModel(BaseModel):
    """
    input an image, return a differentiable loss with respect to the original image
    """

    def __init__(self, clip_name: str, target_text="A image of a bomb", *args, **kwargs):
        self.processor = CLIPProcessor.from_pretrained(clip_name)
        clip = CLIPModel.from_pretrained(clip_name)
        super(TransformerClipAttackLanguageModel, self).__init__(clip, self.processor.tokenizer, None, *args, **kwargs)
        self.clip = clip
        self.target_text = target_text
        # prepare text embedding
        self.target_embedding = self.prepare_text_embedding()
        print(f"finished initializing the model {clip_name}")

    @torch.no_grad()
    def change_target_text(self, target_text: str):
        self.target_text = target_text
        self.target_embedding = self.prepare_target_embedding()

    @torch.no_grad()
    def prepare_target_embedding(self):
        inputs = self.processor(text=[self.target_text], return_tensors="pt", padding=True)
        text_outputs = self.clip.text_model(input_ids=inputs.input_ids.to(self.device))
        text_embeds = text_outputs[1]
        text_embeds = self.clip.text_projection(text_embeds)
        self.target_embedding = text_embeds
        return text_embeds

    def forward(self, x: str) -> Tensor:
        """
        :param x: an input adversarial string
        :return: loss of cosine similarity between target
        """
        inputs = self.processor(text=[x], return_tensors="pt", padding=True)
        text_outputs = self.clip.text_model(input_ids=inputs.input_ids.to(self.device))
        text_embeds = text_outputs[1]
        text_embeds = self.clip.text_projection(text_embeds)
        return self.clip_cosine_similarity(text_embeds)

    def clip_cosine_similarity(self, embeds):
        input_embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
        target_embeds = self.target_embedding / self.target_embedding.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(target_embeds.to(input_embeds.device), input_embeds.t())


class OpenClipAttackLanguageModel(BaseModel):
    """
    input an image, return a differentiable loss with respect to the original image
    """

    def __init__(self, model_name: str, target_text="A image of a bomb", *args, **kwargs):
        clip, _, preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        super(OpenClipAttackLanguageModel, self).__init__(clip, self.tokenizer, None, *args, **kwargs)
        self.clip = clip
        self.target_text = target_text
        # prepare text embedding
        self.target_embedding = self.prepare_target_embedding()
        print(f"finished initializing the model {model_name}")

    @torch.no_grad()
    def change_target_text(self, target_text: str):
        self.target_text = target_text
        self.target_embedding = self.prepare_target_embedding()

    @torch.no_grad()
    def prepare_target_embedding(self):
        text = self.tokenizer([self.target_text])
        text_features = self.clip.encode_text(text.to(self.device)).squeeze()
        text_features /= text_features.norm(dim=-1)  # D
        self.target_embedding = text_features
        return self.target_embedding

    def forward(self, x: str or Tensor or List[str], input_ids: Optional[Tensor] = None) -> Tensor:
        """
        :param x: an input adversarial string or embedding
        :param input_ids:
        :return: loss of cosine similarity between target
        """
        if isinstance(x, str) or isinstance(x, list):
            input_ids = self.tokenizer(x).to(self.device)  # B, L
            x = self.clip.token_embedding(input_ids)
        text_features = self.encode_text(x.to(self.device), input_ids=input_ids, normalize=True)
        return -self.clip_cosine_similarity(text_features)  # minimize - cosine, maximize + cosine

    def clip_cosine_similarity(self, input_embeds):
        input_embeds = input_embeds / input_embeds.norm(p=2, dim=-1, keepdim=True)  # B, D
        text_embeds = self.target_embedding / self.target_embedding.norm(p=2)  # D
        return input_embeds @ text_embeds  # B

    def encode_text(self, x: str or Tensor, input_ids: Optional[Tensor] = None, normalize: bool = True):
        """
        必须是ClipModel类型的。这个是ClipModel的encode_text直接改过来的
        """
        cast_dtype = self.clip.transformer.get_cast_dtype()
        x = x.to(cast_dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.clip.positional_embedding.to(cast_dtype)
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)  # 这个模型只能输入77个词，因此可以预先把attn_mask设定了
        x = self.clip.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, _ = text_global_pool(x, input_ids, self.clip.text_pool_type)
        if self.clip.text_projection is not None:
            if isinstance(self.clip.text_projection, nn.Linear):
                x = self.clip.text_projection(x)
            else:
                x = x @ self.clip.text_projection
        return F.normalize(x, dim=-1) if normalize else x
