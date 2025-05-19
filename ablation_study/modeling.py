import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from probts.model.forecaster import BinConv
class LightningBinConv(BinConv, pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        # assert inputs.shape[-1] == self.context_length
        # assert targets.shape[-1] == self.prediction_length

        # distr_args, loc, scale = self(past_target)
        # distr = self.distr_output.distribution(distr_args, loc, scale)
        # loss = -distr.log_prob(future_target)
        #
        # return loss.mean()
        logits = self(inputs)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        # loss = F.softplus(-targets * logits).mean()
        self.log("train_loss", loss)  # TODO: use log properly
        print(loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
