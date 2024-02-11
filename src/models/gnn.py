import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class WeightedGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(WeightedGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)

        return F.log_softmax(x, dim=1)

# 가중치가 있는 그래프 데이터 생성
num_nodes = 5
x = torch.rand((num_nodes, 3))  # 노드(아이템)의 특징
edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], dtype=torch.long).t().contiguous()
edge_weights = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float)  # 엣지의 가중치

data = Data(x=x, edge_index=edges, edge_attr=edge_weights)

# 모델과 옵티마이저 설정
model = WeightedGCN(num_node_features=3, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 훈련 과정
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    # 예제에서는 실제 레이블을 사용하지 않기 때문에 가상의 레이블을 생성하여 loss를 계산합니다.
    # 실제 사용 시에는 적절한 레이블 데이터를 제공해야 합니다.
    y = torch.randint(0, 2, (data.num_nodes,))
    loss = F.nll_loss(out, y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
