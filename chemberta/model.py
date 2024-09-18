# molecule predictor
import torch
import torch.nn as nn

from transformers import AutoModelForSequenceClassification, GraphormerForGraphClassification, GraphormerModel

from layers import AttentionPooling
from util import get_statistical, zero_index

graphormer_models = [
    'clefourrier/pcqm4mv2_graphormer_base',
]
selected_graphormer_model = graphormer_models[0]

chemberta_models = [
    'seyonec/PubChem10M_SMILES_BPE_450k',
    'DeepChem/ChemBERTa-77M-MTR',
    'seyonec/ChemBERTa-zinc-base-v1',
    'seyonec/ChemBERTa_zinc250k_v2_40k'
    'DeepChem/ChemBERTa-77M-MLM',
]
selected_chemberta_model = chemberta_models[2]


class GraphormerFeatureDecoder(nn.Module):
    def __init__(self, max_len):
        super(GraphormerFeatureDecoder, self).__init__()
        self.pooling = AttentionPooling(input_dim=768, max_len=max_len)  # attention pooling

    def forward(self, x):  # [batch_size, seq_len, feature_dim]
        x = self.pooling(x)  # [batch_size, max_len, feature_dim]
        x = x.view(x.size(0), -1)  # [batch_size, max_len * feature_dim]

        return x


class GraphormerFeatureExtractor(nn.Module):
    def __init__(self, max_len=16):
        super(GraphormerFeatureExtractor, self).__init__()
        self.model = GraphormerForGraphClassification.from_pretrained(
            selected_graphormer_model,
            num_classes=1,  # num_classes for the downstream task
            ignore_mismatched_sizes=True,
        )
        self.encoder = self.model.encoder
        # self.decoder = GraphormerFeatureDecoder(max_len)  # attention pooling
        # self.decoder = get_statistical  # get mean of dim 1
        self.decoder = zero_index  # graphormer에서 기본으로 사용함. 첫번째 logits를 제외하고 전부 제거해 크기 맞춤

    def forward(self,
                input_nodes,
                input_edges,
                attn_bias,
                in_degree,
                out_degree,
                spatial_pos,
                attn_edge_type):
        output = self.model.encoder(
            input_nodes=input_nodes,
            input_edges=input_edges,
            attn_bias=attn_bias,
            in_degree=in_degree,
            out_degree=out_degree,
            spatial_pos=spatial_pos,
            attn_edge_type=attn_edge_type,
        )['last_hidden_state']
        output = self.decoder(output)
        output = output.flatten()

        return output


class ChembertaDecoder(nn.Module):
    def __init__(self, max_len):
        super(ChembertaDecoder, self).__init__()
        self.pooling = AttentionPooling(input_dim=768, max_len=max_len)  # attention pooling

    def forward(self, x):  # [batch_size, seq_len, feature_dim]
        x = self.pooling(x)  # [batch_size, max_len, feature_dim]
        x = x.view(x.size(0), -1)  # [batch_size, max_len * feature_dim]

        return x


class ChembertaFeatureExtractor(nn.Module):
    def __init__(self, max_len) -> None:
        super(ChembertaFeatureExtractor, self).__init__()
        self.chemberta_encoder = AutoModelForSequenceClassification.from_pretrained(
            selected_chemberta_model,
            num_labels=1
        )
        # self.chemberta_decoder = ChembertaDecoder(max_len=max_len)
        self.chemberta_decoder = zero_index

        # <class 'transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification'>
        print(type(self.chemberta_encoder))

        self.encoder = self.chemberta_encoder.roberta

    def forward(self, batch):
        x = self.encoder(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask
        )

        x = x[0]
        # print(x, type(x), x.shape)
        x = self.chemberta_decoder(x)
        # print(len(x))
        # print(x[0].shape)
        x = x.flatten()

        return x


class ChemBERT(nn.Module):
    def __init__(self, fp_dim, out_dim, max_chemberta_len, max_graphormer_len):
        super(ChemBERT, self).__init__()
        self.chemberta_encoder = ChembertaFeatureExtractor(
            max_len=max_chemberta_len
        )
        self.graphormer_encoder = GraphormerFeatureExtractor(
            max_len=max_graphormer_len
        )

        projection_dim = (768 * max_chemberta_len +
                          768 * max_graphormer_len +
                          fp_dim)
        print('projection dim:', projection_dim)

        self.projection = nn.Linear(in_features=projection_dim, out_features=projection_dim)
        self.ln = nn.LayerNorm(normalized_shape=projection_dim)
        self.out = nn.Linear(in_features=projection_dim, out_features=out_dim)

        self.act = nn.GELU()
        self.drop = nn.Dropout(0.144)

    def forward(self, batch):
        enc_out = self.chemberta_encoder(batch)

        # 'input_nodes', 'input_edges', 'attn_bias', 'in_degree', 'out_degree', 'spatial_pos', 'attn_edge_type'
        graphormer_out = self.graphormer_encoder(
            input_nodes=batch.input_nodes,
            input_edges=batch.input_edges,
            attn_bias=batch.attn_bias,
            in_degree=batch.in_degree,
            out_degree=batch.out_degree,
            spatial_pos=batch.spatial_pos,
            attn_edge_type=batch.attn_edge_type,
        )

        fp = batch.fp.squeeze(0)

        # h = torch.concat([enc_out], dim=1)
        # print(enc_out.shape, fp.shape)
        h = torch.concat([enc_out, graphormer_out, fp], dim=0)
        # h = torch.concat([enc_out, batch.mol_f, graphormer_out], dim=1)
        h = self.projection(h)
        h = self.ln(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.out(h)

        return h
