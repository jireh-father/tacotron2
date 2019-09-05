from torch import nn
import torch

class Tacotron2Loss(nn.Module):
    def __init__(self, use_linear_loss, use_loss_reduction_sum):
        super(Tacotron2Loss, self).__init__()
        self.use_linear_loss = use_linear_loss
        reduction= 'mean'
        if use_loss_reduction_sum:
            reduction="sum"
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.linear_loss = nn.L1Loss(reduction=reduction)

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = self.mse_loss(mel_out, mel_target) + self.mse_loss(mel_out_postnet, mel_target)

        gate_loss = self.bce_loss(gate_out, gate_target)
        if self.use_linear_loss:
            linear_loss = self.linear_loss(mel_out, mel_target)
            return mel_loss + gate_loss + linear_loss
        else:
            return mel_loss + gate_loss
