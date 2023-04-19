import torch
import torch.nn as nn

class Add_Norm(nn.Module):

    def __init__(self, normalized_shape) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x = x + residual
        return self.norm(x)

class FeedForward(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_hidden: int = 2048,
        p_dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_hidden),
            nn.SELU(inplace=True),
            nn.Dropout(p_dropout),
            nn.Linear(in_features=d_hidden, out_features=d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class TransformerBlock(nn.Module):

    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        num_heads: int = 8
    ) -> None:
        super().__init__()
        self.mh_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )
        self.addnorm1 = Add_Norm((seq_len, embed_dim))

        self.ffn = FeedForward(d_model=embed_dim)
        self.addnorm2 = Add_Norm((seq_len, embed_dim))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        intermediate, _ = self.mh_attention(query, key, value)
        intermediate = self.addnorm1(intermediate, query)
        out = self.ffn(intermediate)
        out = self.addnorm2(out, intermediate)
        return out

class CA_Block(nn.Module):

    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        num_heads: int = 8
    ) -> None:
        super().__init__()
        self.transformer_block = TransformerBlock(
            embed_dim=embed_dim, seq_len=seq_len, num_heads=num_heads
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer_block(query, key_value, key_value)

class SA_Block(nn.Module):

    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        num_heads: int = 8
    ) -> None:
        super().__init__()
        self.transformer_block = TransformerBlock(
            embed_dim=embed_dim, seq_len=seq_len, num_heads=num_heads
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer_block(x, x, x)

class Add_Norm(nn.Module):

    def __init__(self, normalized_shape) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x = x + residual
        return self.norm(x)



class TransformerBlock_LayerNormBefore(nn.Module):

    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        num_heads: int = 8
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm((seq_len, embed_dim))
        self.mh_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )

        self.addnorm = Add_Norm((seq_len, embed_dim))
        self.ffn = FeedForward(d_model=embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor
    ) -> torch.Tensor:
        query_norm = self.norm(query)
        support_norm = self.norm(support)
        intermediate, _ = self.mh_attention(query_norm, support_norm, support_norm)

        out = self.addnorm(intermediate, query)
        out = self.ffn(out) + intermediate
        return out

class CA_Block_LayerNormBefore(nn.Module):

    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        num_heads: int = 8
    ) -> None:
        super().__init__()
        self.transformer_block = TransformerBlock_LayerNormBefore(
            embed_dim=embed_dim, seq_len=seq_len, num_heads=num_heads
        )

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer_block(query, support)

class SA_Block_LayerNormBefore(nn.Module):

    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        num_heads: int = 8
    ) -> None:
        super().__init__()
        self.transformer_block = TransformerBlock_LayerNormBefore(
            embed_dim=embed_dim, seq_len=seq_len, num_heads=num_heads
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer_block(x, x)
    
        