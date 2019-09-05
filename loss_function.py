from torch import nn
import torch

class Tacotron2Loss(nn.Module):
    def __init__(self, use_linear_loss, reg_weight):
        super(Tacotron2Loss, self).__init__()
        self.use_linear_loss = use_linear_loss
        self.reg_weight = reg_weight

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)

        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        if self.use_linear_loss:
            linear_loss = torch.mean(torch.abs(mel_out_postnet - mel_target))
            return mel_loss + gate_loss + linear_loss
        else:
            return mel_loss + gate_loss
