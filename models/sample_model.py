import torch
import torch.nn as nn


class SampleCNN(nn.Module):
    """
    Minimal model for the compiler pipeline.

    Pipeline:
      Input -> Conv2D -> ReLU -> MaxPool -> Flatten -> FullyConnected
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # For input (N, 1, 28, 28):
        # after conv: (N, 8, 28, 28)
        # after pool: (N, 8, 14, 14)
        self.fc = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_sample_model() -> nn.Module:
    return SampleCNN()


def get_sample_input(batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
    return torch.randn(batch_size, 1, 28, 28, device=device)

