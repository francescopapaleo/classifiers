import torch
import torch.nn as nn
from torchinfo import summary

class ExponentialDilationCNN(nn.Module):
    def __init__(self, num_layers):
        super(ExponentialDilationCNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.layers.append(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=dilation))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def calc_receptive_field(self):
        receptive_field = 1
        for layer in self.layers:
            dilation = layer.dilation[0]
            kernel_size = layer.kernel_size[0]
            receptive_field += (kernel_size - 1) * dilation
        return receptive_field


class LinearDilationCNN(nn.Module):
    def __init__(self, num_layers):
        super(LinearDilationCNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = i + 1
            self.layers.append(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=dilation))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def calc_receptive_field(self):
        receptive_field = 1
        for layer in self.layers:
            dilation = layer.dilation[0]
            kernel_size = layer.kernel_size[0]
            receptive_field += (kernel_size - 1) * dilation
        return receptive_field


if __name__ == "__main__":
    model_exp = ExponentialDilationCNN(num_layers=4)
    model_lin = LinearDilationCNN(num_layers=4)

    dummy_input = torch.randn(1, 1, 512)

    summary_exp = summary(model_exp, input_size=dummy_input.shape, depth=4, verbose=1)
    summary_lin = summary(model_lin, input_size=dummy_input.shape, depth=4, verbose=1)

    print(summary_exp)
    exp_receptive_field = model_exp.calc_receptive_field()
    print(f"Exponential Model Receptive Field: {exp_receptive_field}\n\n")

    print(summary_lin)
    lin_receptive_field = model_lin.calc_receptive_field()
    print(f"Linear Model Receptive Field: {lin_receptive_field}\n\n")

    for i in range(4):
        print(f"Exponential Model Layer {i + 1} - Dilation: {2**i}, Receptive Field: {model_exp.layers[i].dilation[0]}\n")

    for i in range(4):
        print(f"Linear Model Layer {i + 1} - Dilation: {i + 1}, Receptive Field: {model_lin.layers[i].dilation[0]}\n")
