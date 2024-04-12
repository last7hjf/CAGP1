import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, GATConv
import time
import torch.optim as optim
import torchmetrics.functional as MF
import argparse

def perpare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    return args


class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, num_heads)
        self.conv2 = GATConv(h_feats * num_heads, num_classes, 1)
    def forward(self, blocks, features):
        x = F.elu(self.conv1(blocks[0], features).flatten(1))
        x = self.conv2(blocks[1], x).mean(1)
        return x


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)


    def forward(self, blocks, features):
        x = F.relu(self.conv1(blocks[0], features))
        x = self.conv2(blocks[1], x)
        return x


def train(model, g, train_mask, device):
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.2, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_mask[(g.ndata['inner_node'] == 0).nonzero(as_tuple=True)[0]] = False
    train_idx = (g.ndata["train_mask"] == True).nonzero(as_tuple=True)[0]
    train_idx.to(device)
    g.to(device)
    print("number of train node:{}".format(len(train_idx)))
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.DataLoader(
        g, train_idx, sampler, device=device, batch_size=1024
    )

    start_time = time.time()
    f_time = 0
    b_time = 0
    for epoch in range(1, 61):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            f_start = time.time()
            blocks = [b.to(device) for b in blocks]
            input_feat = blocks[0].srcdata["feat"]
            output_label = blocks[-1].dstdata["label"]
            output_logit = model(blocks, input_feat)
            f_end = time.time()
            f_time += (f_end - f_start)
            b_start = time.time()
            loss = criterion(output_logit, output_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b_end = time.time()
            b_time += (b_end - b_start)
            total_loss += loss.item()
            end_time = time.time()
            train_time = end_time - start_time
        if epoch % 20 == 0:
            print("after {0} training rounds ，loss: {1}，total time cost：{2}s".format(epoch, total_loss / (it + 1), train_time))

    print('total time cost：{0} s, forward cost：{1} s, backward cost：{2} s'.format( train_time, f_time, b_time))


@torch.no_grad()
def test(model, g, device, num_classes):
    model.eval()
    ys = []
    y_hats = []
    test_idx = (g.ndata["test_mask"] == True).nonzero(as_tuple=True)[0]
    test_idx.to(device)
    print("number of test node:{}".format(len(test_idx)))
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.DataLoader(
        g, test_idx, sampler, device=device, batch_size=1024, use_uva=True
    )
    for input_nodes, output_nodes, blocks in dataloader:
        blocks = [b.to(device) for b in blocks]
        input_feat = blocks[0].srcdata["feat"]
        ys.append(blocks[-1].dstdata["label"])
        y_hats.append(model(blocks, input_feat))
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys), task='multiclass', num_classes=num_classes)

def main():
    args = perpare()
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(device))
    subdataset = dgl.load_graphs("CAGP_part/{0}_sg{1}.bin".format(args.dataset, args.gpu))
    sg = subdataset[0][0]
    print(sg)
    labels = sg.ndata["label"].to(device)
    num_classes = labels.max().item() + 1
    train_mask = sg.ndata["train_mask"].to(device)
    if args.model == "GAT":
        model = GAT(sg.ndata["feat"].shape[1], 256, num_classes, 8)
        print("use GAT model...")
    elif args.model == "GCN":
        model = GCN(sg.ndata["feat"].shape[1], 256, num_classes)
        print("use GCN model...")
    else:
        raise ValueError("Undefined model: {}".format(args.model))
    model = model.to(device)
    print("train the model...")
    train(model, sg, train_mask, device)
    print("test the model")
    test_acc = test(model, sg, device, num_classes)
    print("test accuracy为{}".format(test_acc.item()))

if __name__ == "__main__":
    main()