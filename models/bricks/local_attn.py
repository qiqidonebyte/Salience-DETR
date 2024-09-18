import torch
import torch.nn as nn

class LocalAttentionWindowModule(nn.Module):
    def __init__(self, min_window_size=33, max_window_size=99):
        super(LocalAttentionWindowModule, self).__init__()
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size

    def forward(self, boxes):
        num_queries = boxes.size(0)
        tgt_size = num_queries

        # 初始化注意力掩码矩阵，填充为 False
        attn_mask = torch.zeros((tgt_size, tgt_size), device=boxes.device, dtype=torch.bool)

        # 为每个目标生成局部注意力掩码
        for i in range(num_queries):
            # 计算局部窗口大小
            window_size = self.calculate_local_window_size(boxes[i][2:])
            start_col = max(i - window_size // 2, 0)
            end_col = min(i + window_size // 2 + 1, tgt_size)

            # 填充局部窗口内的掩码
            attn_mask[i, start_col:end_col] = True

        # 填充对角线，表示每个查询至少关注自己
        torch.fill_diagonal(attn_mask, 1)

        return attn_mask

    def calculate_local_window_size(self, box_size):
        # 这里可以添加任何自定义的逻辑来根据目标尺寸计算窗口大小
        scale_factor = (max(box_size) / min(box_size)) ** 0.5
        window_size = int(self.min_window_size * scale_factor)
        return max(self.min_window_size, min(self.max_window_size, window_size))