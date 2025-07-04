import torch
import torch.nn as nn
import torch.optim as optim
import math
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 模型参数
d_model = 64  # 嵌入维度 / 模型维度
nhead = 4  # 多头注意力的头数
num_layers = 2  # Transformer Encoder 的层数
dim_feedforward = 128  # 前馈网络的隐藏层维度
output_dim = 1  # 预测输出的维度 (例如，下一个点的值，如果是多元预测可以改变)
dropout = 0.1  # Dropout 比率
max_seq_len = 50  # 最大输入序列长度 (根据数据集调整)


class NumericalEmbedding(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        # input_dim: 输入序列中每个元素的特征维度1（因为是单个数值）
        # d_model: Transformer模型的内部维度
        self.linear = nn.Linear(input_dim, d_model)
        self.scale_factor = math.sqrt(d_model)  # 缩放因子，帮助梯度稳定

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, input_dim) -> (batch_size, seq_len, 1)
        # 经过线性层映射到 (batch_size, seq_len, d_model)
        return self.linear(x) * self.scale_factor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)  # 注册为buffer，它不是模型参数，但需要保存状态

    def forward(self, x: torch.Tensor):
        # x: (seq_len, batch_size, d_model)
        # pe: (max_len, 1, d_model)
        # 将位置编码加到输入x上
        # 注意：TransformerEncoderLayer期望的输入是 (seq_len, batch_size, feature_dim)
        # 所以我们这里也保持这种维度顺序
        return x + self.pe[: x.size(0), :]


# 无需单独定义，直接使用 nn.TransformerEncoderLayer
# 它内部包含了 Multi-Head Self-Attention, Feed-Forward Network, Residual Connections, Layer Normalization
# class TransformerBlock(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward, dropout):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.activation = nn.ReLU() # 或 GeLU

#     # 这个 forward 方法是 nn.TransformerEncoderLayer 的简化版
#     def forward(self, src, src_mask=None, src_key_padding_mask=None):
#         src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)

#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src


class PredictionHead(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor):
        # x: (seq_len, batch_size, d_model)
        # 我们通常只需要序列的最后一个时间步的输出进行预测
        # 或者对所有时间步的输出进行池化
        # 这里我们取最后一个时间步的输出
        return self.linear(x[-1, :, :])  # (batch_size, output_dim)


class NumericalTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        output_dim: int,
        dropout: float,
        max_seq_len: int,
    ):
        super().__init__()

        self.input_embedding = NumericalEmbedding(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)  # 在嵌入和PE之后，进入Encoder之前

        # nn.TransformerEncoderLayer 包含了 MultiHeadAttention, FeedForward, Add&Norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # 重要: PyTorch 默认是 (seq_len, batch_size, feature_dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.prediction_head = PredictionHead(d_model, output_dim)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
    ):
        # src: (batch_size, seq_len, input_dim) -> e.g., (32, 10, 1)

        # 1. Input Embedding
        # Convert to (batch_size, seq_len, d_model)
        src = self.input_embedding(src)

        # 2. Transpose for Transformer Encoder (to seq_len, batch_size, d_model)
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, d_model)

        # 3. Positional Encoding
        src = self.positional_encoding(src)

        # 4. Dropout before Transformer Encoder
        src = self.dropout(src)

        # 5. Transformer Encoder
        # src_mask: (seq_len, seq_len) - 用于因果 L-R 注意力, 防止看到未来信息
        # src_key_padding_mask: (batch_size, seq_len) - 用于处理变长序列的attention masking
        output = self.transformer_encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        # output: (seq_len, batch_size, d_model)

        # 6. Prediction Head
        # output needs to be (batch_size, output_dim)
        final_prediction = self.prediction_head(output)

        return final_prediction


# 辅助函数: 创建因果注意力掩码 (上三角矩阵为负无穷)
def generate_causal_mask(seq_len: int):
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


from sklearn.preprocessing import MinMaxScaler


def create_synthetic_time_series_dataset(
    num_samples: int, seq_len: int, prediction_len: int = 1
):
    """
    生成一个简单的正弦波时间序列数据集。
    每个样本的输入是seq_len个连续点，输出是接下来的prediction_len个点。
    """
    total_len = num_samples + seq_len
    time = np.linspace(0, 100, total_len)
    data = np.sin(time) + np.random.randn(total_len) * 0.1  # 加入一点噪声

    # 归一化数据
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    samples = []
    labels = []
    for i in range(num_samples):
        if i + seq_len + prediction_len > total_len:
            break
        input_seq = data[i : i + seq_len]
        output_val = data[
            i + seq_len : i + seq_len + prediction_len
        ]  # 预测下一个prediction_len个值
        samples.append(input_seq)
        labels.append(output_val)

    X = np.array(samples)
    y = np.array(labels)

    # 确保X是 (batch_size, seq_len, 1)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    # 确保y是 (batch_size, prediction_len)
    y = y.reshape(y.shape[0], prediction_len)

    return X, y, scaler  # 返回scaler以便后续逆变换查看真实值


# 生成数据集
num_samples = 2000  # 样本数量
max_seq_len = 50  # 输入序列长度
output_dim = 1  # 预测下一个点的值

X_data, y_data, data_scaler = create_synthetic_time_series_dataset(
    num_samples, max_seq_len, output_dim
)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

# 转换为 PyTorch Tensor 和 DataLoader
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# 初始化模型
input_dim = 1  # 每个时间步的特征数量 (这里是单个数值)
model = NumericalTransformer(
    input_dim=input_dim,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dim_feedforward=dim_feedforward,
    output_dim=output_dim,
    dropout=dropout,
    max_seq_len=max_seq_len,
).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()  # 均方误差，适用于回归任务
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            # PyTorch TransformerEncoderLayer 默认 batch_first=False
            # 所以输入应该是 (seq_len, batch_size, feature_dim)
            # 我们的DataLoader输出是 (batch_size, seq_len, feature_dim)
            # 模型内部的NumericalEmbedding已经将 (batch_size, seq_len, 1) -> (batch_size, seq_len, d_model)
            # 并在送入TransformerEncoder前 permute 到 (seq_len, batch_size, d_model)

            # 生成因果注意力掩码，防止模型在预测时“偷看”未来的信息
            # mask: (seq_len, seq_len), 只有当前及之前的元素可见
            # 对于Encoder，如果目标是预测下一个 token，需要因果掩码。
            # 但在这里，我们的目标是预测整个序列的下一个值，而非逐个预测序列中的所有值，
            # 因此，对于Encoder的自注意力部分，通常不需要因果掩码（除非你想限制每个位置只能关注前面的信息）。
            # 因为我们的数据是 (X_seq, Y_next_value)，X_seq 作为一个整体来看待。
            # 如果是生成序列，Decoder部分才严格需要因果掩码。
            # 对于Encoder，通常不加mask，让每个元素看到所有历史信息。
            # 但是，为了更贴近“预测下一个值”的思想，以及如果未来你扩展到生成序列，使用mask是个好习惯。
            # 这里的 mask 应该针对 src 序列的 seq_len。
            # 如果你的问题是预测整个序列的下一个值，且输入是整个历史序列，你可以不使用 src_mask。
            # 但如果输入序列中的每个元素都应该只能关注它之前的元素来计算自己的表示，则需要mask。
            # 对于时间序列预测，通常允许看到整个历史序列。这里我们不使用 src_mask。

            # src_mask = generate_causal_mask(data.size(1)).to(device) # data.size(1) 是 seq_len
            # output = model(data, src_mask=src_mask) # 传入mask

            # 简化：对于这种"序列到下一个预测值"的任务，Encoder通常无需causal mask
            output = model(data)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )  # 梯度裁剪，防止梯度爆炸
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.6f}")
    return train_losses


# 评估函数
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            # output = model(data, src_mask=generate_causal_mask(data.size(1)).to(device))
            output = model(data)  # 评估时也不用 causal mask
            loss = criterion(output, target)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Test Loss: {avg_loss:.6f}")
    return avg_loss


print("\nStarting training...")
num_epochs = 50  # 训练轮数，简单数据集可以少一些
train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs)

print("\nStarting evaluation...")
test_loss = evaluate_model(model, test_loader, criterion)
