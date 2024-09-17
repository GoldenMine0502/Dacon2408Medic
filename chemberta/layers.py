import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, max_len):
        """
        input_dim: 입력 텐서의 차원 (feature_dim)
        max_len: 고정된 출력 시퀀스 길이
        """
        super(AttentionPooling, self).__init__()
        self.max_len = max_len

        # 학습 가능한 query 벡터 (고정된 길이의 max_len 개의 query 벡터)
        self.query = nn.Parameter(torch.randn(max_len, input_dim))

        # 선형 변환
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = input_dim ** 0.5  # 어텐션 스케일링을 위한 상수

    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, input_dim]
        mask: [batch_size, seq_len] (optional) 패딩 마스크
        """
        batch_size, seq_len, input_dim = x.shape

        # Query, Key, Value 계산
        Q = self.query.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, max_len, input_dim]
        K = self.key(x)  # [batch_size, seq_len, input_dim]
        V = self.value(x)  # [batch_size, seq_len, input_dim]

        # 어텐션 스코어 계산 (QK^T / sqrt(d_k))
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, max_len, seq_len]

        # 마스크가 있다면 패딩된 부분에 큰 음수 값을 부여하여 소프트맥스에서 0에 가까운 값이 되도록 처리
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        # 어텐션 가중치 계산
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, max_len, seq_len]

        # 어텐션 가중치를 사용하여 Value 가중합
        context = torch.matmul(attn_weights, V)  # [batch_size, max_len, input_dim]

        return context


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5  # 어텐션 스케일링을 위한 상수

    def forward(self, x, mask):
        # Q, K, V 계산
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 어텐션 스코어 계산 (Q와 K의 내적)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, seq_len, seq_len]

        # 마스크 적용: 패딩된 부분에 큰 음수 값을 넣어 소프트맥스에서 0에 가까운 값이 되도록 만듦
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        # 소프트맥스 적용하여 어텐션 가중치 계산
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, seq_len, seq_len]

        # 어텐션 가중치를 사용해 V 값 가중합
        context = torch.matmul(attn_weights, V)  # [batch_size, seq_len, hidden_dim]

        return context, attn_weights