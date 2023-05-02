from Transformer.transformer_blocks import CA_Block_LayerNormBefore as CA_Block, SA_Block_LayerNormBefore as SA_Block
from CNN.backbone import ResNet_Backbone

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Deep_ResNet_Transformer(nn.Module):

    def __init__(self) -> None:
        super(Deep_ResNet_Transformer, self).__init__()

        self.encoder = ResNet_Backbone()

        # self.pos_encoding = Parameter(
        #     data = torch.randn(1,1024,256),
        #     requires_grad = True
        # )

        self.image_token = Parameter(
            data = torch.randn(1,1,256),
            requires_grad = True
        )

        self.sa1a = SA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.sa1b = SA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.ca1a = CA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.ca1b = CA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.sa2a = SA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.sa2b = SA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.ca2a = CA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.ca2b = CA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.sa3a = SA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.sa3b = SA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.ca3a = CA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.ca3b = CA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.sa4a = SA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.sa4b = SA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.ca4a = CA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.ca4b = CA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.sa5a = SA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.sa5b = SA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.ca5 = CA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.sa6 = SA_Block(seq_len=1025, embed_dim=256, num_heads=4)

        self.mlp1 = nn.Sequential(nn.Linear(in_features=256, out_features=128, bias=True),
                                 nn.SELU(inplace=True))
        self.mlp2 = nn.Sequential(nn.Linear(in_features=128, out_features=64, bias=True),
                                 nn.SELU(inplace=True))
        self.mlp3 = nn.Sequential(nn.Linear(in_features=64, out_features=3, bias=True),
                                 nn.Identity(inplace=True))

        # self._init_weights()

    # def _init_weights(self):
    #     r"""Initiate parameters in the transformer model."""
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)

    def forward(self, batch: dict) -> torch.Tensor:
        # print(f"live :\n{batch['rgb0'][0]}")
        # print(f"bottleneck :\n{batch['rgb1'][0]}")
        encoded_live = self.encoder(batch["rgb0"], batch["vmap0"]).permute(0, 2, 3, 1).reshape(-1, 1024, 256)
        encoded_bottleneck = self.encoder(batch["rgb1"], batch["vmap1"]).permute(0, 2, 3, 1).reshape(-1, 1024, 256)
        encoded_live = encoded_live #+ self.pos_encoding
        encoded_bottleneck = encoded_bottleneck #+ self.pos_encoding

        encoded_live = torch.concat((encoded_live, self.image_token.repeat(encoded_live.shape[0],1,1)), dim=1)
        encoded_bottleneck = torch.concat((encoded_bottleneck, self.image_token.repeat(encoded_live.shape[0],1,1)), dim=1)

        print(f"Encoded live:\n{encoded_live}")
        print(f"\nEncoded bottleneck:\n{encoded_bottleneck}")

        attention_live =self.sa1a(encoded_live)
        attention_bottleneck =self.sa1b(encoded_bottleneck)
        attention_live2 =self.ca1a(attention_live, attention_bottleneck)
        attention_bottleneck2 =self.ca1b(attention_bottleneck, attention_live)
        attention_live2 =self.sa2a(attention_live2)
        attention_bottleneck2 =self.sa2b(attention_bottleneck2)
        attention_live3 =self.ca2a(attention_live2, attention_bottleneck2)
        attention_bottleneck3 =self.ca2b(attention_bottleneck2, attention_live2)
        attention_live3 =self.sa3a(attention_live3)
        attention_bottleneck3 =self.sa3b(attention_bottleneck3)
        attention_live4 =self.ca3a(attention_live3, attention_bottleneck3)
        attention_bottleneck4 =self.ca3b(attention_bottleneck3, attention_live3)
        attention_live4 =self.sa4a(attention_live4)
        attention_bottleneck4 =self.sa4b(attention_bottleneck4)
        attention_live5 =self.ca4a(attention_live4, attention_bottleneck4)
        attention_bottleneck5 =self.ca4b(attention_bottleneck4, attention_live4)
        attention_live5 =self.sa5a(attention_live5)
        attention_bottleneck5 =self.sa5b(attention_bottleneck5)
        attention =self.ca5(attention_live5, attention_bottleneck5)
        attention =self.sa6(attention)

        print(f"\nAttention:\n{attention}")

        # out = torch.mean(attention, dim=1) #mean of image tokens
        out = attention[:,-1,:]
        print(f"\nAveraged attention:\n{out}")
        out = self.mlp1(out)
        out = self.mlp2(out)
        out = self.mlp3(out)

        batch['pred'] = out
        return out


if __name__ == "__main__":
    rand_live = torch.randint(low=0, high=255, size=(8, 3, 256, 256)).to(dtype=torch.float32)
    rand_bottle = torch.randint(low=0, high=255, size=(8, 3, 256, 256)).to(dtype=torch.float32)
    # rand_live = torch.randint(low=-1, high=1, size=(8, 3, 128, 128))
    # rand_bottle = torch.randint(low=-1, high=1, size=(8, 3, 128, 128))
    # print(rand_live)
    # print(rand_bottle)

    batch = {
        "rgb0": rand_live,
        "vmap0": rand_live,
        "rgb1": rand_bottle,
        "vmap1": rand_bottle,
    }

    model = Deep_ResNet_Transformer()
    print("Parameter cound: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    out = model(batch)
    print(out.shape)
    # print(out)




