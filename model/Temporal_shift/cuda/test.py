import torch
from shift import Shift

def test_shift_layer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化Shift层，假设输入通道是3，步长stride=1
    shift_layer = Shift(channel=3, stride=1).to(device)

    # 构造输入tensor，shape: [batch, channel, height, width]
    # 模拟骨架坐标图卷积中可能的输入格式
    input_tensor = torch.randn(2, 3, 5, 5, device=device, requires_grad=True)

    # 前向传播
    output = shift_layer(input_tensor)

    print("Output tensor:")
    print(output)

    # 构造一个简单的loss函数，反向传播测试
    loss = output.sum()
    loss.backward()

    print("Gradient wrt input:")
    print(input_tensor.grad)

    print("Gradient wrt xpos:")
    print(shift_layer.xpos.grad)

    print("Gradient wrt ypos:")
    print(shift_layer.ypos.grad)


if __name__ == "__main__":
    test_shift_layer()
