from pathlib import Path

import torch

from _old_strcture.signature_net.models import SignatureNet1


def determine_params(pretrained_model):
    """
    Determine the number of trainable params in a PyTorch model
    :param pretrained_model: path
    :return: int
    """
    model = SignatureNet1(num_classes=13, is_constrained=False).to(torch.device('cpu'))
    state_dict = torch.load(pretrained_model, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model_state_dict'])

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Total params            : {total_params}')
    print(f'Total trainable params  : {total_trainable_params}')


if __name__ == '__main__':
    # pre-trained model path
    determine_params(
        pretrained_model=
        Path(r'/scratch/p288722/runtime_data/scd_pytorch/18_models_hcal_non_homo_1/fold_1/signature_net_brands.pt')
    )
