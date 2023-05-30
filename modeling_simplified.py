import numpy as np
import torch
import copy
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
from transformers import CLIPConfig, WhisperConfig, LlamaConfig
from transformers import CLIPModel, LlamaForCausalLM, WhisperModel
from transformers.utils import logging

logger = logging.get_logger(__name__)


class MM_LLMs(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.temporal_position_embeddings = nn.Embedding(
            config.n_frames, 
            config.image_config.projection_dim
        )

        self.image_encoder = CLIPModel(config.image_config)
        self.video_encoder = CLIPModel(config.image_config)
        self.audio_encoder = WhisperModel(config.audio_config)
        self.llm = LlamaForCausalLM(config.llm_config)

        attn_dropout = 0.1
        is_add_bias_kv = True
        is_add_zero_attn = True
        self.temporal_self_attention = nn.MultiheadAttention(
            config.image_config.projection_dim, 
            config.attention_heads,
            dropout=attn_dropout,
            add_bias_kv=is_add_bias_kv,
            add_zero_attn=is_add_zero_attn
        )

        self.video_align_attention = nn.MultiheadAttention(
            config.llm_config.hidden_size, 
            config.attention_heads,
            dropout=attn_dropout,
            add_bias_kv=is_add_bias_kv,
            add_zero_attn=is_add_zero_attn
        )

        self.audio_align_attention = nn.MultiheadAttention(  
            config.llm_config.hidden_size, 
            config.attention_heads,
            dropout=attn_dropout,
            add_bias_kv=is_add_bias_kv,
            add_zero_attn=is_add_zero_attn
        )

        self.image_align_attention = nn.MultiheadAttention(
            config.llm_config.hidden_size, 
            config.attention_heads,
            dropout=attn_dropout,
            add_bias_kv=is_add_bias_kv,
            add_zero_attn=is_add_zero_attn
        )

        self.transform_video_to_hidden = nn.Linear(config.image_config.projection_dim, 
                                                   config.llm_config.hidden_size)
        self.transform_audio_to_hidden = nn.Linear(config.audio_config.d_model, 
                                                   config.llm_config.hidden_size)
        self.transform_image_to_hidden = nn.Linear(config.image_config.projection_dim, 
                                                   config.llm_config.hidden_size)
        
        self.project_image = nn.Conv1d(
            config.image_config.projection_dim,
            config.image_config.projection_dim, 
            kernel_size=48,
            stride=36
        )
        self.project_video = nn.Conv1d(
            config.image_config.projection_dim,
            config.image_config.projection_dim, 
            kernel_size=36,
            stride=30
        )
        self.project_audio = nn.Conv1d(
            config.audio_config.d_model,
            config.audio_config.d_model, 
            kernel_size=240,
            stride=220
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.layer_norm = nn.LayerNorm(config.image_config.projection_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.loss_fct = CrossEntropyLoss()
        self.init_weights()

    def forward(self, inputs=None):
        # """
        # :param inputs:
        #             video_frames: (B x F)
        #             audios: B x 1
        #             images: B x 1
        #             input_ids: B x L
        #             labels: B x L
        #
        # :return: loss when training else None
        # """
        text_embeddings, attention_mask, labels = self.prepare_inputs_for_generation(inputs)

        if 'inference' in inputs and inputs['inference'] is True:
            # generate_ids = self.llm.generate(input_ids=inputs['input_ids'], inputs_embeds=text_embeddings, max_new_tokens=128)
            generate_ids = self.llm.generate(inputs_embeds=text_embeddings, max_new_tokens=128)

            return generate_ids
        outputs = self.llm(inputs_embeds=text_embeddings, attention_mask=attention_mask, labels=labels)

        return outputs

    def prepare_inputs_for_generation(self, inputs):

        image_features = self.encode_image(inputs['images']) if inputs['images'] is not None else None
        audio_features = self.encode_audio(inputs['audios']) if inputs['audios'] is not None else None
        video_features = self.encode_video(inputs['videos']) if inputs['videos'] is not None else None
        # embed_tokens = self.llm.model.model.embed_tokens
        embed_tokens = self.llm.model.embed_tokens
        text_embeddings = embed_tokens(inputs['input_ids'])

        token_embeddings = embed_tokens.weight.unsqueeze(0).repeat(
            text_embeddings.size(0), 1, 1).transpose(0, 1)

        ingore_num = 0
        if video_features is not None:
            video_starts = embed_tokens(inputs['video_starts']).unsqueeze(1)
            video_ends = embed_tokens(inputs['video_ends']).unsqueeze(1)

            # video_features = self.project_video(video_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

            video_features = self.transform_video_to_hidden(video_features)
            
            video_features = self.video_align_attention(
                video_features.transpose(0, 1), 
                token_embeddings,
                token_embeddings)[0].transpose(0, 1).contiguous()

            video_inputs = torch.cat([torch.cat([video_starts, video_features], dim=1), video_ends], dim=1)

            text_embeddings = torch.cat([torch.cat([text_embeddings[:, 0, :].unsqueeze(1), video_inputs], dim=1), text_embeddings[:, 1:, :]], dim=1)

            ingore_num += (video_inputs.size(1))

        if audio_features is not None:
            audio_starts = embed_tokens(inputs['audio_starts']).unsqueeze(1)
            audio_ends = embed_tokens(inputs['audio_ends']).unsqueeze(1)
            
            audio_features = self.project_audio(audio_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

            audio_features = self.transform_audio_to_hidden(audio_features)
            
            # mean pooling
            # audio_features = torch.sum(audio_features, dim=1) / audio_features.size(1) 
            # audio_features = audio_features.unsqueeze(1)

            audio_features = self.video_align_attention(audio_features.transpose(0, 1), token_embeddings, token_embeddings)[0].transpose(0, 1).contiguous()

            audio_inputs = torch.cat([torch.cat([audio_starts, audio_features], dim=1), audio_ends], dim=1)

            text_embeddings = torch.cat(
                [torch.cat([text_embeddings[:, 0, :].unsqueeze(1), audio_inputs], dim=1), text_embeddings[:, 1:, :]],
                dim=1)

            ingore_num += (audio_inputs.size(1))

        if image_features is not None:
            image_starts = embed_tokens(inputs['image_starts']).unsqueeze(1)
            image_ends = embed_tokens(inputs['image_ends']).unsqueeze(1)

            image_features = self.project_image(image_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

            image_features = self.transform_image_to_hidden(image_features)
            image_features = self.video_align_attention(
                image_features.transpose(0, 1),
                token_embeddings, token_embeddings)[0].transpose(0, 1).contiguous()

            image_inputs = torch.cat([torch.cat([image_starts, image_features], dim=1), image_ends], dim=1)

            text_embeddings = torch.cat(
                [torch.cat([text_embeddings[:, 0, :].unsqueeze(1), image_inputs], dim=1), text_embeddings[:, 1:, :]],
                dim=1
            )

            ingore_num += (image_inputs.size(1))

        if 'attention_mask' in inputs:
            attention_mask = torch.tensor([1]*ingore_num*text_embeddings.size(0), device=text_embeddings.device).view(text_embeddings.size(0), -1)
            attention_mask = torch.cat([attention_mask, inputs['attention_mask']], dim=1)
        else:
            attention_mask = None

        if 'labels' in inputs and inputs['labels'] is not None:
            labels = torch.tensor([-100]*ingore_num*text_embeddings.size(0), device=text_embeddings.device).view(text_embeddings.size(0), -1)
            labels = torch.cat([labels, inputs['labels']], dim=1)
        else:
            labels = None

        return text_embeddings, attention_mask, labels

    def encode_video(self, videos):
        # simple image encoding without temporal embedding and self attention
        videos = videos.view(-1, videos.size(-3), videos.size(-2), videos.size(-1))
        video_outputs = self.video_encoder.get_image_features(videos)
        video_features = video_outputs
        temporal_pos = torch.tensor(
            [[i for i in range(self.config.n_frames)] 
             for j in range(videos.size(0) // self.config.n_frames)],
            dtype=torch.int, device=video_features.device).view(-1)

        frame_temporal_pos_embed = self.temporal_position_embeddings(temporal_pos)

        video_features = (video_features + frame_temporal_pos_embed).view(videos.size(0) // self.config.n_frames,
                                                                          self.config.n_frames, -1)

        video_features = video_features.transpose(0, 1).contiguous()
        self_attn_video_features = self.temporal_self_attention(video_features, video_features, video_features)[0]

        return self_attn_video_features.transpose(0, 1).contiguous()
    
    def encode_video_long(self, videos):
        # simple image encoding without temporal embedding and self attention
        videos = videos.view(-1, videos.size(-3), videos.size(-2), videos.size(-1))
        video_features = self.video_encoder.visual_projection(self.video_encoder.vision_model(videos)[0])[:, 1:, :]
        video_features = video_features.reshape(
            videos.size(0) // self.config.n_frames,
            self.config.n_frames * video_features.size(1), -1).contiguous()
        
        return video_features

    def encode_audio(self, audios):
        audio_features = self.audio_encoder.encoder(audios)
        return audio_features[0]

    def encode_image(self, images):
        # vision_outputs = self.image_encoder.get_image_features(images)
        # image_features = vision_outputs  # pooled_output
        # image_features = self.visual_projection(pooled_output)
        # image_features = image_features.unsqueeze(1)
        
        image_features = self.image_encoder.visual_projection(self.image_encoder.vision_model(images)[0])[:, 1:, :]
        return image_features


class MM_LLMs_Config(PretrainedConfig):
    model_type = 'mm_llms'
    is_composition = True

    def __init__(
        self,
        n_frames=6,
        attention_heads=8,
        clip_config=None,
        whisper_config=None,
        llm_config=None,
        **kwargs
    ):
        self.image_config = clip_config
        self.audio_config = whisper_config
        self.llm_config = llm_config
        self.n_frames = n_frames
        self.attention_heads = attention_heads

        self.hidden_size = max(
            llm_config.hidden_size,
            clip_config.projection_dim,
            whisper_config.d_model,
            clip_config.projection_dim
        )

        super().__init__(**kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["image_config"] = self.image_config.to_dict()
        output["audio_config"] = self.audio_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['n_frames'] = self.n_frames
        output['attention_heads'] = self.attention_heads
        output['hidden_size'] = self.hidden_size
        output["model_type"] = self.__class__.model_type
        return output
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        clip_config = CLIPConfig.from_dict(config_dict['image_config'])
        whisper_config = WhisperConfig.from_dict(config_dict['audio_config'])
        llm_config = LlamaConfig.from_dict(config_dict['llm_config'])

        return cls(clip_config=clip_config, whisper_config=whisper_config, llm_config=llm_config, **kwargs)
