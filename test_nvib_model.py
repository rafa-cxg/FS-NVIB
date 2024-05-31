import torch

from models.nvib_vision_transformer import NvibVisionTransformer as nvib_vit
from models.vision_transformer import VisionTransformer as vit

# Initialise the model

# To get this to work you need to not initialise the weight of the models
# self.apply(self._init_weights)

delta=0
alpha_tau=-100
stdev_tau=0
nvib_layers = [11]

# Set seed
torch.manual_seed(0)
model = vit(
    img_size=[224],
    patch_size=16,
    in_chans=3,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    qkv_bias=True,
    representation_size=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    norm_layer=torch.nn.LayerNorm,
    ape=False,
    patch_norm=True,
    use_checkpoint=False,
)
torch.manual_seed(0)
model_nvib = nvib_vit(
    img_size=[224],
    patch_size=16,
    in_chans=3,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    qkv_bias=True,
    representation_size=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    norm_layer=torch.nn.LayerNorm,
    ape=False,
    patch_norm=True,
    use_checkpoint=False,
    delta=delta,
    alpha_tau=alpha_tau,
    stdev_tau=stdev_tau,
    nvib_layers = nvib_layers,
)

from models.nvib_vision_transformer import vit_small as nvib_vit
from models.vision_transformer import vit_small as vit

# Dino model
url_dino = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
state_dict_dino = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url_dino)

# Dino weights for small model
dino_vitsmall = vit(patch_size=16, num_classes=0)
dino_vitsmall.load_state_dict(state_dict_dino, strict=False)

# Dino weights for small model NVIB
dino_vitsmall_nvib = nvib_vit(patch_size=16, num_classes=0,
                              delta=delta,
                                alpha_tau=alpha_tau,
                                stdev_tau=stdev_tau,
                                nvib_layers = nvib_layers
                                )
dino_vitsmall_nvib.load_state_dict(state_dict_dino, strict=False)

#Deit model
url_deit = "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
state_dict_deit = torch.hub.load_state_dict_from_url(url=url_deit)["model"]
for k in ['head.weight', 'head.bias']:
            if k in state_dict_deit:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict_deit[k]

# Deit weights for small model
deit_vitsmall = vit(patch_size=16, num_classes=0)
deit_vitsmall.load_state_dict(state_dict_deit, strict=False)


# Deit weights for small model NVIB
deit_vitsmall_nvib = nvib_vit(patch_size=16, num_classes=0, delta=delta,
    alpha_tau=alpha_tau,
    stdev_tau=stdev_tau,
    nvib_layers = nvib_layers,)
deit_vitsmall_nvib.load_state_dict(state_dict_deit, strict=False)


# Test base model
def test_base_model_train():
    # Random input that is torch.Size([25, 3, 224, 224])
    x = torch.randn(2, 3, 224, 224)

    # Forward pass
    model.train()
    y = model(x)

    # Forward pass
    model_nvib.train()
    y_nvib = model_nvib(x)
    
    # Check average closeness
    print("Average closeness: ", torch.mean(torch.abs(y - y_nvib[0])))
    print("Average closeness CLS: ", torch.mean(torch.abs(y[:,0,:] - y_nvib[0][:,0,:])))

    # check equality
    assert torch.allclose(y, y_nvib[0], atol=1e-4), "Models are not equal"

    print("Base model test passed")

def test_dino_vitsmall_train():
    # Random input that is torch.Size([25, 3, 224, 224])
    x = torch.randn(2, 3, 224, 224)

    # Forward pass
    dino_vitsmall.train()
    y = dino_vitsmall(x)

    # Forward pass
    dino_vitsmall_nvib.train()
    y_nvib = dino_vitsmall_nvib(x)
 
    print("Average closeness: ", torch.mean(torch.abs(y - y_nvib[0])))

    # check equality
    # assert torch.allclose(y, y_nvib[0], atol=1e-4), "Models are not equal"

    #print("Dino model test passed")

# Test deit_vitsmall
def test_deit_vitsmall_train():
    # Random input that is torch.Size([25, 3, 224, 224])
    x = torch.randn(2, 3, 224, 224)

    # Forward pass
    deit_vitsmall.train()
    y = deit_vitsmall(x)

    # Forward pass
    deit_vitsmall_nvib.train()
    y_nvib = deit_vitsmall_nvib(x)

    print("Average closeness: ", torch.mean(torch.abs(y - y_nvib[0])))

    # check equality
    #assert torch.allclose(y, y_nvib[0], atol=1e-4), "Models are not equal"

    #print("Deit model test passed")


# eval tests
def test_base_model_eval():
    # Random input that is torch.Size([25, 3, 224, 224])
    x = torch.randn(2, 3, 224, 224)

    # Forward pass
    model.eval()
    y = model(x)

    # Forward pass
    model_nvib.eval()
    y_nvib = model_nvib(x)

    # check equality
    assert torch.allclose(y, y_nvib[0], atol=1e-4), "Models are not equal"

    print("Base model test passed")

def test_dino_vitsmall_eval():
    # Random input that is torch.Size([25, 3, 224, 224])
    x = torch.randn(2, 3, 224, 224)

    # Forward pass
    dino_vitsmall.eval()
    y = dino_vitsmall(x)

    # Forward pass
    dino_vitsmall_nvib.eval()
    y_nvib = dino_vitsmall_nvib(x)

    print("Average closeness: ", torch.mean(torch.abs(y - y_nvib[0])))

    # check equality
    # assert torch.allclose(y, y_nvib[0], atol=1e-4), "Models are not equal"

    # print("Dino model test passed")

# Test deit_vitsmall
def test_deit_vitsmall_eval():
    # Random input that is torch.Size([25, 3, 224, 224])
    x = torch.randn(2, 3, 224, 224)

    # Forward pass
    deit_vitsmall.eval()
    y = deit_vitsmall(x)

    # Forward pass
    deit_vitsmall_nvib.eval()
    y_nvib = deit_vitsmall_nvib(x)

    print("Average closeness: ", torch.mean(torch.abs(y - y_nvib[0])))

    # check equality
    # assert torch.allclose(y, y_nvib[0], atol=1e-4), "Models are not equal"
    # print("Deit model test passed")


def main():

    # test_base_model_train()
    # test_base_model_eval()
    
    # The pretrained models are trained to look at images.
    # When given random input their embeddings are large
    
    # This doesn't appear to be the problem. Embeddings are large even with the correct input 
    
    test_dino_vitsmall_train()
    test_deit_vitsmall_train()


    test_dino_vitsmall_eval()
    test_deit_vitsmall_eval()


if __name__ == '__main__':
    main()