import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
import logging
from tqdm import tqdm

from baselines.multimodal_transformer.transformer.tvc_dataset import TVCaptionDataset
from torch.utils.data import DataLoader
from baselines.multimodal_transformer.transformer.tvc_dataset import \
    caption_collate, prepare_batch_inputs
logger = logging.getLogger(__name__)


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        self.log_softmax = nn.LogSoftmax(dim=-1)

        smoothing_value = label_smoothing / (tgt_vocab_size - 1)  # count for the ground-truth word
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        # one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size, with indices in [-1, tgt_vocab_size-1], `-1` is ignored
        """
        valid_indices = target != self.ignore_index  # ignore examples with target value -1
        target = target[valid_indices]
        output = self.log_softmax(output[valid_indices])

        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(output, model_prob, reduction="sum")


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT"s gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionEncoding(nn.Module):
    """
    Add positional information to input tensor.

    """

    def __init__(self, n_filters=128, max_len=500):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(PositionEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = self.pe.data[:x.size(-2), :]  # (#x.size(-2), n_filters)
        extra_dim = len(x.size()) - 2
        for _ in range(extra_dim):
            pe = pe.unsqueeze(0)
        x = x + pe
        return x


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states, attention_mask):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, another_tensor, attention_mask, is_cross=False):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        if is_cross and another_tensor is not None:
            self_output = self.self(input_tensor, another_tensor, another_tensor, attention_mask)
        else:
            self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FusionBlock(nn.Module):
    def __init__(self, config):
        super(FusionBlock, self).__init__()
        self.config = config
        self.linear = nn.Linear(config.max_cap_len*2, config.max_cap_len)
        self.dropout = nn.Dropout(config.hidden_dropout_pro)
        self.activation = nn.ReLU()

    def forward(self, dec_enc_v, dec_enc_s):
        fusion = torch.cat((dec_enc_v, dec_enc_s))
        fusion = self.linear(fusion)


class BertEmbeddingsWithVideo(nn.Module):
    """Construct the embeddings from word (+ video), position and token_type embeddings.
    input_ids (batch_size, sequence_length), with [1, sequence_length_1 + 1] filled with [VID]
    video_features (batch_size, sequence_length),
    with [1, sequence_length_1 + 1] as real features, others as zeros
    ==> video features and word embeddings are merged together by summing up.
    """
    def __init__(self, config, add_postion_embeddings=True):
        super(BertEmbeddingsWithVideo, self).__init__()
        """add_postion_embeddings: whether to add absolute positional embeddings"""
        self.add_postion_embeddings = add_postion_embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.word_vec_size, padding_idx=0)
        self.word_fc = nn.Sequential(
            BertLayerNorm(config.word_vec_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.word_vec_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.video_embeddings = nn.Sequential(
            BertLayerNorm(config.video_feature_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.video_feature_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.position_embeddings = PositionEncoding(n_filters=config.hidden_size,
                                                    max_len=config.max_position_embeddings)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def set_pretrained_embedding(self, pretrained_embedding, freeze=True):
        """Note the from_pretrained does not work in-place, so you need to assign value to the embedding"""
        assert pretrained_embedding.shape == self.word_embeddings.weight.shape  # ensure equal shape
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze,
                                                            padding_idx=self.word_embeddings.padding_idx)

    def get_caption_word_embedding(self, caption_input_ids):
        """ text_input_ids: (N, Lt) """
        words_embeddings = self.word_fc(self.word_embeddings(caption_input_ids))  # (N, Lt, D)
        words_embeddings = self.position_embeddings(words_embeddings)
        return words_embeddings  # (N, Lt, D)

    def forward(self, video_feature, vid_token_type_ids, sub_ids, word_token_type_ids):

        words_embeddings = self.word_fc(self.word_embeddings(sub_ids))
        video_embeddings = self.video_embeddings(video_feature)
        word_token_type_embeddings = self.token_type_embeddings(word_token_type_ids)
        vid_token_type_embeddings = self.token_type_embeddings(vid_token_type_ids)

        sub_embeddings = words_embeddings + word_token_type_embeddings
        vid_embeddings = video_embeddings + vid_token_type_embeddings

        """
        sep = vid_embeddings.shape[1]
        ctx_embeddings = torch.cat((vid_embeddings, sub_embeddings), 1)
        ctx_embeddings = self.position_embeddings(ctx_embeddings)
        n_vid_embeddings, n_sub_embeddings = torch.split(ctx_embeddings, [sep, ctx_embeddings.shape[1]-sep], dim=1)
        assert n_vid_embeddings.shape[1] == vid_embeddings.shape[1], "The split is not run properly"
        assert n_sub_embeddings.shape[1] == sub_embeddings.shape[1], "The split is not run properly"
        """

        sub_embeddings = self.dropout(self.LayerNorm(self.position_embeddings(sub_embeddings)))
        vid_embeddings = self.dropout(self.LayerNorm(self.position_embeddings(vid_embeddings)))

        return vid_embeddings, sub_embeddings


# ===================================
# Original encoder (selfAttn) - One layer
# ===================================
class BertLayerNoMemoryUntied(nn.Module):
    def __init__(self, config):
        super(BertLayerNoMemoryUntied, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, diagonal_mask=False):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property.
        Returns:

        """
        self_attention_mask = attention_mask.unsqueeze(1)
        if diagonal_mask:  # mask subsequent words
            max_len = hidden_states.size(1)
            self_attention_mask = self_attention_mask * \
                torch.tril(self_attention_mask.new_ones(max_len, max_len), diagonal=0)
        attention_output = self.attention(hidden_states, None, self_attention_mask)  # (N, L, D)
        intermediate_output = self.hidden_intermediate(attention_output)  # (N, L, D)
        layer_output = self.output(intermediate_output, attention_output)  # (N, L, D)
        return layer_output


# ===================================
# Dual streams (crossAttn) - One layer
# ===================================
class CrossBertLayerNoMemoryUnited(nn.Module):
    def __init__(self, config):
        super(CrossBertLayerNoMemoryUnited, self).__init__()
        self.config = config
        self.self_attn_v = BertAttention(config)
        self.self_attn_s = BertAttention(config)
        self.cross_attention_v = BertAttention(config)
        self.cross_attention_s = BertAttention(config)
        self.hidden_intermediate_v = BertIntermediate(config)
        self.hidden_intermediate_s = BertIntermediate(config)
        self.output_v = BertOutput(config)
        self.output_s = BertOutput(config)

    def forward(self, hidden_states_v, hidden_states_s, attention_mask_v, attention_mask_s, diagonal_mask=False):

        attention_mask_v = attention_mask_v.unsqueeze(1)
        attention_mask_s = attention_mask_s.unsqueeze(1)
        attention_output_v = self.self_attn_v(hidden_states_v, None, attention_mask_v)
        attention_output_s = self.self_attn_s(hidden_states_s, None, attention_mask_s)
        attention_output_v = self.cross_attention_v(attention_output_v, attention_output_s,
                                                    attention_mask_s, is_cross=True)  # (N, L, D)
        attention_output_s = self.cross_attention_s(attention_output_s, attention_output_v,
                                                    attention_mask_v, is_cross=True)
        intermediate_output_v = self.hidden_intermediate_v(attention_output_v)  # (N, L, D)
        intermediate_output_s = self.hidden_intermediate_s(attention_output_s)
        layer_output_v = self.output_v(intermediate_output_v, attention_output_v)  # (N, L, D)
        layer_output_s = self.output_s(intermediate_output_s, attention_output_s)
        return layer_output_v, layer_output_s


# ===================================
# Encoder - Wrapper
# ===================================
class BertEncoderNoMemoryUntied(nn.Module):
    def __init__(self, config):
        super(BertEncoderNoMemoryUntied, self).__init__()
        self.model_name = config.model_name
        self.config = config
        self.v_stream = None
        self.s_stream = None
        self.former_stream = None
        self.latter_stream = None
        self.stream_assign()

    def forward(self, sub, sub_mask, vid, vid_mask, attention_mask, diagonal_mask=False,
                output_all_encoded_layers=True):
        all_vid_stream = []
        all_sub_stream = []
        all_enc_layers = []
        if self.model_name in ["2streams_self", "2streams_dec"]:
            for layer_idx, (layer_module_v, layer_module_s) in enumerate(zip(self.v_stream, self.s_stream)):
                vid = layer_module_v(vid, vid_mask, diagonal_mask)
                sub = layer_module_s(sub, sub_mask, diagonal_mask)
                if output_all_encoded_layers:
                    all_vid_stream.append(vid)
                    all_sub_stream.append(sub)
                    all_enc_layers.append(torch.cat((vid, sub), 1))
            if not output_all_encoded_layers:
                all_vid_stream.append(vid)
                all_sub_stream.append(sub)
                all_enc_layers.append(torch.cat((vid, sub), 1))

            if self.model_name == "2streams_self":
                return all_enc_layers
            else:
                return all_vid_stream, all_sub_stream

        elif self.model_name in ["2to1stream_self"]:
            for layer_idx, (layer_module_v, layer_module_s) in enumerate(zip(self.v_stream, self.s_stream)):
                vid = layer_module_v(vid, vid_mask, diagonal_mask)
                sub = layer_module_s(sub, sub_mask, diagonal_mask)
                if output_all_encoded_layers:
                    all_enc_layers.append(torch.cat((vid, sub), 1))
            hidden_state = torch.cat((vid, sub), 1)
            for layer_idx, layer_module in enumerate(self.latter_stream):
                hidden_state = layer_module(hidden_state, attention_mask, diagonal_mask)
                if output_all_encoded_layers:
                    all_enc_layers.append(hidden_state)
            if not output_all_encoded_layers:
                all_enc_layers.append(hidden_state)

            return all_enc_layers

        elif self.model_name in ["2to1stream_cross"]:
            for layer_idx, layer_module in enumerate(self.former_stream):
                vid, sub = layer_module(vid, sub, vid_mask, sub_mask, diagonal_mask)
                if output_all_encoded_layers:
                    all_enc_layers.append(torch.cat((vid, sub), 1))
            hidden_state = torch.cat((vid, sub), 1)
            for layer_idx, layer_module in enumerate(self.latter_stream):
                hidden_state = layer_module(hidden_state, attention_mask, diagonal_mask)
                if output_all_encoded_layers:
                    all_enc_layers.append(hidden_state)
            if not output_all_encoded_layers:
                all_enc_layers.append(hidden_state)

            return all_enc_layers

        elif self.model_name in ["2streams_cross"]:
            for layer_idx, layer_module in enumerate(self.former_stream):
                vid, sub = layer_module(vid, sub, vid_mask, sub_mask)
                if output_all_encoded_layers:
                    all_enc_layers.append(torch.cat((vid, sub), 1))
            if not output_all_encoded_layers:
                all_enc_layers.append(torch.cat((vid, sub), 1))
            return all_enc_layers

        else:
            assert NameError, "The input model name is not supported"

    def stream_assign(self):
        num_former_layers = self.config.num_hidden_layers // 2
        num_latter_layers = self.config.num_hidden_layers - num_former_layers
        if self.model_name in ["2streams_self", "2streams_dec"]:
            self.v_stream = nn.ModuleList([BertLayerNoMemoryUntied(self.config)
                                           for _ in range(self.config.num_hidden_layers)])
            self.s_stream = nn.ModuleList([BertLayerNoMemoryUntied(self.config)
                                           for _ in range(self.config.num_hidden_layers)])
        elif self.model_name in ["2to1stream_self"]:
            self.v_stream = nn.ModuleList([BertLayerNoMemoryUntied(self.config) for _ in range(num_former_layers)])
            self.s_stream = nn.ModuleList([BertLayerNoMemoryUntied(self.config) for _ in range(num_former_layers)])
            self.latter_stream = nn.ModuleList([BertLayerNoMemoryUntied(self.config) for _ in range(num_latter_layers)])
        elif self.model_name in ["2to1stream_cross"]:
            self.former_stream = nn.ModuleList([CrossBertLayerNoMemoryUnited(self.config)
                                               for _ in range(num_former_layers)])
            self.latter_stream = nn.ModuleList([BertLayerNoMemoryUntied(self.config)
                                               for _ in range(num_latter_layers)])
        elif self.model_name in ["2streams_cross"]:
            self.former_stream = nn.ModuleList(CrossBertLayerNoMemoryUnited(self.config) for _ in range(self.config.num_hidden_layers))
        else:
            assert NameError, "The input model name is not supported"


# ===================================
# Original decoder - One layer
# ===================================
class BertDecoderLayerNoMemoryUntied(nn.Module):
    def __init__(self, config):
        super(BertDecoderLayerNoMemoryUntied, self).__init__()
        self.config = config
        self.self_attention = BertSelfAttention(config)
        self.norm1 = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dec_enc_attention = BertSelfAttention(config)
        self.norm2 = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output = BertOutput(config)  # linear + residual + layernorm

    def forward(self, dec_hidden_states, dec_mask, enc_outputs, enc_mask, diagonal_mask=True):
        """
        Args:
            dec_hidden_states: (N, Lt, D)
            dec_mask: (N, Lt)
            enc_outputs: (N, Lv, D)
            enc_mask: (N, Lv)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property.
        Returns:

        """
        self_attention_mask = dec_mask.unsqueeze(1)
        if diagonal_mask:  # mask subsequent words
            max_len = dec_mask.size(1)  # Lt
            self_attention_mask = self_attention_mask * \
                torch.tril(self_attention_mask.new_ones(max_len, max_len), diagonal=0)

        # 1, dec self attn + add_norm
        attention_output = self.self_attention(
            dec_hidden_states, dec_hidden_states, dec_hidden_states, self_attention_mask)  # (N, Lt, D)
        attention_output = self.norm1(attention_output + dec_hidden_states)  # (N, Lt, D)

        # 2, dec enc attn + add_norm
        # Is the attention mask correct?
        # Yes! Use the mask associated with key/value, not query. (query, key, value)
        # Additionally, there is no need to do subsequent masking, since each word has the right to see
        # all the video info.
        dec_enc_attention_output = self.dec_enc_attention(
            attention_output, enc_outputs, enc_outputs, enc_mask.unsqueeze(1))  # (N, Lt, D)
        dec_enc_attention_output = self.norm2(attention_output + dec_enc_attention_output)  # (N, Lt, D)

        # 3, linear + add_norm
        dec_enc_attention_output = self.output(dec_enc_attention_output, dec_enc_attention_output)  # (N, Lt, D)
        return dec_enc_attention_output  # (N, Lt, D)


# =================================
# Cross decoder - One layer
# =================================
class CrossBertDecoderLayerNoMemoryUntied(nn.Module):
    def __init__(self, config):
        super(CrossBertDecoderLayerNoMemoryUntied, self).__init__()
        self.config = config
        self.self_attention = BertSelfAttention(config)
        self.norm1 = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dec_enc_attention_v = BertSelfAttention(config)
        self.norm2 = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dec_enc_attention_s = BertSelfAttention(config)
        self.norm3 = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.bridge =
        self.output = BertOutput(config)

    def forward(self, dec_hidden_states, dec_mask, enc_output_v, enc_mask_v,
                enc_output_s, enc_mask_s, diagonal_mask=True):
        self_attention_mask = dec_mask.unsqueeze(1)
        if diagonal_mask:  # mask subsequent words
            max_len = dec_mask.size(1)  # Lt
            self_attention_mask = self_attention_mask * \
                                  torch.tril(self_attention_mask.new_ones(max_len, max_len), diagonal=0)

        # 1. dec self attn + add_norm
        attention_output = self.self_attention(
            dec_hidden_states, dec_hidden_states, dec_hidden_states, self_attention_mask)
        attention_output = self.norm1(attention_output + dec_hidden_states)


        # 2.1 enc_attn from video side
        dec_enc_attention_output_v = self.dec_enc_attention_v(
            attention_output, enc_output_v, enc_output_v, enc_mask_v.unsqueeze(1))
        dec_enc_attention_output_v = self.norm2(attention_output+dec_enc_attention_output_v)

        # 2.2 enc_attn from sub side
        dec_enc_attention_output_s = self.dec_enc_attention_s(
            attention_output, enc_output_s, enc_output_s, enc_mask_s.unsqueeze(1)
        )
        dec_enc_attention_output_s = self.norm3(attention_output+dec_enc_attention_output_s)

        # 3. fusion_block
        dec_enc_attention_output = dec_enc_attention_output_v + dec_enc_attention_output_s
        """
        1. sum
        2. concat + linear
        """

        # 4. linear + add_norm
        dec_enc_attention_output = self.output(dec_enc_attention_output, dec_enc_attention_output)
        return dec_enc_attention_output


# ===================================
# Decoder - Wrapper
# ===================================
class BertDecoderNoMemoryUntied(nn.Module):
    def __init__(self, config):
        super(BertDecoderNoMemoryUntied, self).__init__()
        self.layer = nn.ModuleList([BertDecoderLayerNoMemoryUntied(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, dec_hidden_states, dec_mask, enc_outputs, enc_mask,
                diagonal_mask=True, output_all_encoded_layers=False):
        """
        Args:
            dec_hidden_states: (N, Lt, D)
            dec_mask: (N, Lt)
            enc_outputs: (N, Lv, D)
            enc_mask: (N, Lv)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property
            output_all_encoded_layers:

        Returns:

        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            dec_hidden_states = layer_module(
                dec_hidden_states, dec_mask, enc_outputs, enc_mask, diagonal_mask=diagonal_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(dec_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(dec_hidden_states)
        return all_encoder_layers


# ===================================
# Cross decoder - Wrapper
# ===================================
class CrossBertDecoderNoMemoryUntied(nn.Module):
    def __init__(self, config):
        super(CrossBertDecoderNoMemoryUntied, self).__init__()
        self.layer = nn.ModuleList([CrossBertDecoderLayerNoMemoryUntied(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, dec_hidden_states, dec_mask, enc_output_v, enc_mask_v, enc_output_s, enc_mask_s,
                diagonal_mask=True, output_all_encoded_layers=False):
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            dec_hidden_states = layer_module(
                dec_hidden_states, dec_mask, enc_output_v, enc_mask_v,
                enc_output_s, enc_mask_s, diagonal_mask=diagonal_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(dec_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(dec_hidden_states)
        return all_encoder_layers


# ===================================
# Loss calculation and
# Prediction distribution
# ===================================
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = gelu
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        """(N, L, D)"""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights=None):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        if config.share_wd_cls_weight:
            assert bert_model_embedding_weights is not None, \
                "bert_model_embedding_weights should not be None " \
                "when setting --share_wd_cls_weight flag to be true"
            assert config.hidden_size == bert_model_embedding_weights.size(1), \
                "hidden size has be the same as word embedding size when " \
                "sharing word embedding weight and classifier weight"
            self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                     bert_model_embedding_weights.size(0),
                                     bias=False)
            self.decoder.weight = bert_model_embedding_weights
        else:
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        """(N, L, D)"""
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states  # (N, L, vocab_size)


# ===================================
# Model
# ===================================
class MMT(nn.Module):
    def __init__(self, config):
        super(MMT, self).__init__()
        self.config = config
        self.model_name = config.model_name
        self.embeddings = BertEmbeddingsWithVideo(config)
        self.encoder = BertEncoderNoMemoryUntied(config)
        self.decoder = CrossBertDecoderNoMemoryUntied(config) if self.model_name == "2streams_dec" \
            else BertDecoderNoMemoryUntied(config)
        decoder_classifier_weight = self.embeddings.word_embeddings.weight \
            if self.config.share_wd_cls_weight else None
        self.decoder_classifier = BertLMPredictionHead(config, decoder_classifier_weight)
        self.loss_func = LabelSmoothingLoss(config.label_smoothing, config.vocab_size, ignore_index=-1) \
            if "label_smoothing" in config and config.label_smoothing > 0 else nn.CrossEntropyLoss(ignore_index=-1)
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def encode(self, sub_ids, sub_mask, sub_token_type_ids, vid_feature, vid_mask, vid_token_type_ids, ctx_input_mask):
        """
        Args:
            ctx_input_ids: (N, Lctx)
            ctx_input_mask: (N, Lctx)  with 1 indicates valid bits
            ctx_token_type_ids: (N, Lctx) with 0 indicates [VID]
            video_feature: (N, Lctx, D)
        """
        vid_embedding, sub_embedding = self.embeddings(vid_feature, vid_token_type_ids, sub_ids, sub_token_type_ids)

        if self.model_name == "2streams_dec":
            enc_output_v, enc_output_s = self.encoder(sub_embedding, sub_mask, vid_embedding, vid_mask, ctx_input_mask, diagonal_mask=False)  # (N, Lctx, D)
            return enc_output_v[-1], enc_output_s[-1]
        else:
            output = self.encoder(sub_embedding, sub_mask, vid_embedding, vid_mask, ctx_input_mask, diagonal_mask=False)
            return output[-1]

    def decode_lf(self, text_input_ids, text_masks, vid_output, vid_mask, sub_output,  sub_mask, text_input_labels=None):
        """
        Args:
            text_input_ids: (N, Lt)
            text_masks: (N, Lt)  with 1 indicates valid bits
            text_input_labels: (N, Lt)  with `-1` on ignored positions
            encoder_outputs: (N, Lctx, D)
            encoder_masks: (N, Lctx)
        """
        text_embeddings = self.embeddings.get_caption_word_embedding(text_input_ids)  # (N, Lt, D)
        decoder_outputs = self.decoder(
            text_embeddings, text_masks, vid_output, vid_mask, sub_output, sub_mask, diagonal_mask=True)[-1]  # (N, Lt, D)
        prediction_scores = self.decoder_classifier(decoder_outputs)  # (N, Lt, vocab_size)
        caption_loss = 0.
        if text_input_labels is not None:
            caption_loss = self.loss_func(prediction_scores.view(-1, self.config.vocab_size),
                                          text_input_labels.view(-1))
        return caption_loss, prediction_scores

    def decode(self, text_input_ids, text_masks, encoder_outputs, encoder_masks, text_input_labels=None):
        """
        Args:
            text_input_ids: (N, Lt)
            text_masks: (N, Lt)  with 1 indicates valid bits
            text_input_labels: (N, Lt)  with `-1` on ignored positions
            encoder_outputs: (N, Lctx, D)
            encoder_masks: (N, Lctx)
        """
        text_embeddings = self.embeddings.get_caption_word_embedding(text_input_ids)  # (N, Lt, D)
        decoder_outputs = self.decoder(
            text_embeddings, text_masks, encoder_outputs, encoder_masks, diagonal_mask=True)[-1]  # (N, Lt, D)
        prediction_scores = self.decoder_classifier(decoder_outputs)  # (N, Lt, vocab_size)
        caption_loss = 0.
        if text_input_labels is not None:
            caption_loss = self.loss_func(prediction_scores.view(-1, self.config.vocab_size),
                                          text_input_labels.view(-1))
        return caption_loss, prediction_scores

    def forward(self, caption_input_ids, caption_mask, caption_labels,
                sub_ids, sub_mask, sub_token_type_ids,
                video_feature, video_mask, video_token_type_ids, ctx_input_mask, ctx_tokens):
        """
        Args:
            ctx_input_ids: (N, Lctx)  with 1 indicates valid bits
            ctx_input_mask: (N, Lctx)
            ctx_token_type_ids: (N, Lctx)
            video_feature: (N, Lctx, D)
            caption_input_ids: (N, Lt)
            caption_mask: (N, Lt)  with 1 indicates valid bits
            caption_labels: (N, Lt)  with `-1` on ignored positions
        """
        if self.model_name == "2streams_dec":
            vid_output, sub_output = self.encode(sub_ids, sub_mask, sub_token_type_ids,
                                                 video_feature, video_mask, video_token_type_ids,
                                                 ctx_input_mask)  # (N, Lv, D)
            caption_loss, prediction_scores = self.decode_lf(caption_input_ids, caption_mask,
                                                             vid_output, video_mask,
                                                             sub_output, sub_mask,
                                                             text_input_labels=caption_labels)
        else:
            enc_output = self.encode(sub_ids, sub_mask, sub_token_type_ids,
                                 video_feature, video_mask, video_token_type_ids,
                                 ctx_input_mask)
            caption_loss, prediction_scores = self.decode(caption_input_ids, caption_mask,
                                                          enc_output, ctx_input_mask,
                                                          text_input_labels=caption_labels)
        return caption_loss, prediction_scores


# remind me of what the configs are
base_config = edict(
    hidden_size=768,
    vocab_size=None,  # get from word2idx
    video_feature_size=2048,
    max_position_embeddings=None,  # get from max_seq_len
    type_vocab_size=2,
    layer_norm_eps=1e-12,  # bert layernorm
    hidden_dropout_prob=0.1,  # applies everywhere except attention
    num_hidden_layers=2,  # number of transformer layers
    attention_probs_dropout_prob=0.1,  # applies only to self attention
    intermediate_size=768,  # after each self attention
    num_attention_heads=12,
    memory_dropout_prob=0.1
)

"""
########################################################################
train_dataset = TVCaptionDataset(
        ctx_mode="video_sub",
        data_ratio=1,
        data_path="/home/feib/project_test/Cross-Att-TVCaption/data/tvc_train_release.jsonl",
        sub_meta_path="/home/feib/project_test/Cross-Att-TVCaption/data/tvqa_preprocessed_subtitles.jsonl",
        vid_h5_path_or_handler="/home/feib/project_test/Cross-Att-TVCaption/data/tvc_feature_release/video_feature/tvr_i3d_rgb600_avg_cl-1.5.h5",
        word2idx_path="/home/feib/project_test/Cross-Att-TVCaption/cache/tvc_word2idx.json",
        max_cap_len=20,
        max_sub_len=30,
        max_v_len=20,
        h5driver="core",
        clip_length=1.5,
        normalize_vfeat=False,
        is_eval=False
    )
train_loader = DataLoader(train_dataset,
                              collate_fn=caption_collate,
                              batch_size=1,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=False)
vocab_size = len(train_dataset.word2idx)
rt_config = edict(
        hidden_size=768,
        intermediate_size=768,  # after each self attention
        vocab_size=vocab_size,  # get from word2idx
        word_vec_size=300,
        video_feature_size=1024,
        max_position_embeddings=50,  # get from max_seq_len
        type_vocab_size=2,
        layer_norm_eps=1e-12,  # bert layernorm
        hidden_dropout_prob=0.1,  # applies everywhere except attention
        num_hidden_layers=2,  # number of transformer layers
        num_attention_heads=12,
        attention_probs_dropout_prob=0.1,  # applies only to self attention
        initializer_range=0.02,
        label_smoothing=0.1,
        share_wd_cls_weight=False,
        model_name="2streams_dec"
    )


model = MMT(rt_config)
model.to("cuda:1")

for batch_idx, batch in tqdm(enumerate(train_loader)):
    data = prepare_batch_inputs(batch[0],device="cuda:1")
    vid_feat = data["video_feature"]
    vid_mask = data["video_mask"]
    vid_token_type_ids = data["video_token_type_ids"]
    sub_ids = data["sub_ids"]
    sub_mask = data["sub_mask"]
    sub_token_type_ids = data["sub_token_type_ids"]
    #print(vid_feat.shape, vid_mask.shape, vid_token_type_ids.shape)
    #print(sub_ids.shape, sub_mask.shape, sub_token_type_ids.shape)
    loss, pred_scores = model(**data)
    print(loss, pred_scores)
    break
"""
"""
video-sub
-------------
shape of output from encoder: (50, 768)

video
------------
shape of output from encoder: (22, 768)

sub
------------
shape of output from encoder: (32, 768)
"""