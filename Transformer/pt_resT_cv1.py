import os
from pathlib import Path

from Transformer.transformer_blocks import CA_Block_LayerNormBefore as CA_Block, SA_Block_LayerNormBefore as SA_Block
from CNN.backbone import ResNet_Backbone

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class PreTrained_RNTransformer_Class(nn.Module):

    def __init__(self, n_classes: int = 18) -> None:
        super(PreTrained_RNTransformer_Class, self).__init__()

        self.encoder = ResNet_Backbone()

        self.pos_encoding = Parameter(
            data = torch.randn(1,1024,256) /2,
            requires_grad = True
        )

        self.image_token = Parameter(
            data = torch.randn(1,1,256) /2 ,
            requires_grad = True
        )

        self.sa1a = SA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.sa1b = SA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.ca1a = CA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.ca1b = CA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.sa2a = SA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.sa2b = SA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.ca2 = CA_Block(seq_len=1025, embed_dim=256, num_heads=4)
        self.sa3 = SA_Block(seq_len=1025, embed_dim=256, num_heads=4)

        self.mlp1 = nn.Sequential(nn.Linear(in_features=256, out_features=128, bias=True),
                                 nn.SELU(inplace=True))
        self.mlp2 = nn.Sequential(nn.Linear(in_features=128, out_features=64, bias=True),
                                 nn.SELU(inplace=True))
        self.mlp3 = nn.Sequential(nn.Linear(in_features=64, out_features=n_classes, bias=True),
                                 nn.Identity(inplace=True))

        self.initialise_pretrained("pretrained_dres_1/version_0/checkpoints/epoch=2-val_loss=1.8490.ckpt")
        
    def initialise_pretrained(self, ckpt_dir):
        ckpt_dir = os.path.join(Path(__file__).parent.parent, 'logs', ckpt_dir)
        state_dict = torch.load(ckpt_dir, map_location='cpu')['state_dict']
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if "encoder" in name:
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                print(f"Loading {name} parameters")
                name = name[6:] #remove 'model.'
                own_state[name].copy_(param)

    def forward(self, batch: dict) -> torch.Tensor:
        # print(f"live :\n{batch['rgb0'][0]}")
        # print(f"bottleneck :\n{batch['rgb1'][0]}")
        encoded_live = self.encoder(batch["rgb0"], batch["vmap0"]).permute(0, 2, 3, 1).reshape(-1, 1024, 256)
        encoded_bottleneck = self.encoder(batch["rgb1"], batch["vmap1"]).permute(0, 2, 3, 1).reshape(-1, 1024, 256)
        encoded_live = encoded_live + self.pos_encoding
        encoded_bottleneck = encoded_bottleneck + self.pos_encoding

        encoded_live = torch.concat((encoded_live, self.image_token.repeat(encoded_live.shape[0],1,1)), dim=1)
        encoded_bottleneck = torch.concat((encoded_bottleneck, self.image_token.repeat(encoded_live.shape[0],1,1)), dim=1)

        # print(f"Encoded live:\n{encoded_live}")
        # print(f"\nEncoded bottleneck:\n{encoded_bottleneck}")

        attention_live =self.sa1a(encoded_live)
        attention_bottleneck =self.sa1b(encoded_bottleneck)
        attention_live2 =self.ca1a(attention_live, attention_bottleneck)
        attention_bottleneck2 =self.ca1b(attention_bottleneck, attention_live)
        attention_live2 =self.sa2a(attention_live2)
        attention_bottleneck2 =self.sa2b(attention_bottleneck2)
        attention =self.ca2(attention_live2, attention_bottleneck2)
        attention =self.sa3(attention)

        # print(f"\nAttention:\n{attention}")

        # out = torch.mean(attention, dim=1) #mean of image tokens
        out = attention[:,-1,:]
        # print(f"\nAveraged attention:\n{out}")
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

    model = PreTrained_RNTransformer_Class()
    print("Parameter cound: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    with torch.no_grad():
        out = model(batch)
    print(out.shape)
    # print(out)









