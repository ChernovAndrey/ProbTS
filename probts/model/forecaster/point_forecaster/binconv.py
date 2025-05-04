import torch
import torch.nn as nn
from probts.model.forecaster import Forecaster
import torch.nn.functional as F
def sliding_window_batch(x, L, H):
    """
    x: Tensor of shape (B, L+H, C)
    Returns: Tensor of shape (B, H, L, C)
    """
    B, total_len, C = x.shape
    assert total_len >= L + H, "Not enough sequence length for given L and H"

    windows = [x[:, h:h + L, :].unsqueeze(1) for h in range(H)]  # list of (B, 1, L, C)
    return torch.cat(windows, dim=1)  # (B, H, L, C)

def most_probable_monotonic_sequence(p: torch.Tensor):
    """
    p: Tensor of shape (B, D) where each row is a probability vector
    Returns:
        best_sequences: Tensor of shape (B, D) with most probable [1...1, 0...0] sequence per batch
        best_probs: Tensor of shape (B,) with normalized probability of each best sequence
    """
    B, D = p.shape

    # Compute cumulative product of p and (1 - p)
    left_cumprod = torch.cumprod(p, dim=1)  # shape (B, D)
    right_cumprod = torch.cumprod((1 - p).flip(dims=[1]), dim=1).flip(dims=[1])  # shape (B, D)

    # Pad left with 1 at the beginning (per batch)
    ones = torch.ones((B, 1), dtype=p.dtype, device=p.device)
    left = torch.cat([ones, left_cumprod[:, :-1]], dim=1)  # shape (B, D)
    right = right_cumprod  # shape (B, D)

    # Element-wise multiply
    probs = left * right  # shape (B, D)

    # Normalize
    probs_sum = probs.sum(dim=1, keepdim=True)  # shape (B, 1)
    probs_normalized = probs / probs_sum

    # Find best cut index per batch
    best_k = torch.argmax(probs_normalized, dim=1)  # shape (B,)

    # Construct best sequences
    arange = torch.arange(D, device=p.device).unsqueeze(0)  # shape (1, D)
    best_k_expanded = best_k.unsqueeze(1)  # shape (B, 1)
    best_sequences = (arange < best_k_expanded).to(p.dtype)  # shape (B, D)

    # Get best normalized probabilities
    best_probs = torch.gather(probs_normalized, dim=1, index=best_k.unsqueeze(1)).squeeze(1)  # shape (B,)

    return best_sequences, best_probs


class BinConv(Forecaster):
    def __init__(self, context_length: int, num_bins: int, kernel_size_across_bins_2d: int = 3,
                 kernel_size_across_bins_1d: int = 3, num_filters_2d: int = 8,
                 num_filters_1d: int = 32, is_cum_sum: bool = False, **kwargs) -> None:
        """
        Initialize the model with parameters.
        """
        super().__init__(context_length=context_length, **kwargs)
        # Initialize model parameters here
        self.context_length = context_length
        self.num_bins = num_bins
        self.num_filters_2d = num_filters_2d
        self.num_filters_1d = num_filters_1d
        self.kernel_size_across_bins_2d = kernel_size_across_bins_2d
        self.kernel_size_across_bins_1d = kernel_size_across_bins_1d
        self.is_cum_sum = is_cum_sum

        # Conv2d over (context_length, num_bins)
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.num_filters_2d,
            kernel_size=(context_length, kernel_size_across_bins_2d),
            bias=True
        )

        self.conv1d_1 = nn.Conv1d(
            in_channels=self.num_filters_2d,
            out_channels=self.num_filters_1d,
            kernel_size=kernel_size_across_bins_1d,
            bias=True
        )

        self.conv1d_2 = nn.Conv1d(
            in_channels=self.num_filters_1d,
            out_channels=self.num_bins,
            kernel_size=kernel_size_across_bins_1d,
            bias=True
        )

    # def forward(self, inputs):
    #     """
    #     Forward pass for the model.
    #
    #     Parameters:
    #     inputs [Tensor]: Input tensor for the model.
    #
    #     Returns:
    #     Tensor: Output tensor.
    #     """
    #     # Perform the forward pass of the model
    #     return outputs

    def forward(self, x):
        def pad_channels(tensor, pad_size: int, pad_val_left=1.0, pad_val_right=0.0):
            if pad_size == 0:
                return tensor
            left = torch.full((*tensor.shape[:-1], pad_size), pad_val_left, device=tensor.device)
            right = torch.full((*tensor.shape[:-1], pad_size), pad_val_right, device=tensor.device)
            return torch.cat([left, tensor, right], dim=-1)

        x = x.float()
        # x: (batch_size, context_length, num_bins)
        batch_size, context_length, num_bins = x.shape
        assert context_length == self.context_length, "Mismatch in context length"

        pad2d = self.kernel_size_across_bins_2d // 2 if self.kernel_size_across_bins_2d > 1 else 0
        x_padded = pad_channels(x, pad2d)
        x_conv_in = x_padded.unsqueeze(1)
        conv_out = F.relu(self.conv(x_conv_in).squeeze(2))  # (batch_size, num_filters_2d, num_bins)

        pad1d = self.kernel_size_across_bins_1d // 2 if self.kernel_size_across_bins_1d > 1 else 0
        h_padded = pad_channels(conv_out, pad1d)
        h = F.relu(self.conv1d_1(h_padded))

        h_padded = pad_channels(h, pad1d)
        out = self.conv1d_2(h_padded).mean(dim=1)  # (batch_size, num_bins)

        if self.is_cum_sum:
            out = torch.flip(torch.cumsum(torch.flip(out, dims=[1]), dim=1), dims=[1])
        return out

    def loss(self, batch_data):
        """
        Compute the loss for the given batch data.

        Parameters:
        batch_data [dict]: Dictionary containing input data and possibly target data.

        Returns:
        Tensor: Computed loss.
        """
        # Extract inputs and targets from batch_data

        inputs = self.get_inputs(batch_data, 'all')
        # print(f'bool:{torch.allclose(inputs[:, -self.prediction_length:, :], batch_data.future_target_cdf.float())}')
        inputs = sliding_window_batch(inputs, self.context_length, self.prediction_length).float()
        outputs = self(inputs.view(-1, *inputs.shape[2:]))
        # outputs = outputs[:, -self.prediction_length-1:-1, ...]
        target = batch_data.future_target_cdf.float()
        loss = F.binary_cross_entropy_with_logits(input=outputs, target=target.view(-1, *target.shape[2:]), )
        return loss

    def forecast(self, batch_data, num_samples=None):
        inputs = self.get_inputs(batch_data, 'encode')
        current_context = inputs.clone()
        forecasts = []
        for _ in range(self.prediction_length):
            pred = F.sigmoid(self(current_context))  # (B, D)
            # pred = (pred >= 0.5).int()
            pred, _ = most_probable_monotonic_sequence(pred)
            pred = pred.int()
            forecasts.append(pred.unsqueeze(1))  # (B, 1, D)
            next_input = pred.unsqueeze(1)
            current_context = torch.cat([current_context[:, 1:], next_input], dim=1)

        return torch.cat(forecasts, dim=1)  # (B, T, D)
