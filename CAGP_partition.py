import os
os.environ["DGLBACKEND"] = "pytorch"
import numpy as np
import random
import dgl
import torch
import time
import argparse

def perpare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cophy")
    parser.add_argument("--edgelimit", type=float, default="0.63")
    parser.add_argument("--T", type=float, default="500")
    parser.add_argument("--delta", type=float, default="0.8")
    parser.add_argument("--gamma", type=float, default="0.2")
    parser.add_argument("--alpha", type=float, default="0.5")
    parser.add_argument("--beta", type=float, default="1")
    parser.add_argument("--m", type=float, default="1.25")
    parser.add_argument("--lambda1", type=float, default="0.3")
    args = parser.parse_args()
    return args


def gini_torch(x):
    x = x.view(-1)
    x = x.sort()[0]
    n = x.size(0)
    cumx = x.cumsum(0)
    gini = (n + 1 - 2 * torch.sum(cumx) / cumx[-1]) / n
    return gini.item()

def edge_nor_entroy(degrees):
    degrees = degrees.numpy()
    H_en = 0
    for d in degrees:
        H_en += -(d / np.sum(degrees)) * np.log(d / np.sum(degrees))
    return H_en * (1 / np.log(degrees.size))

def vertex_edge(nx_g, vid):
    in_edges = list(nx_g.in_edges(vid))
    out_edges = list(nx_g.out_edges(vid))
    edges = in_edges + out_edges
    return set(edges)

def vertex_neighbor(nx_g, vid):
    neighbor = set(nx_g.neighbors(vid))
    return neighbor

def heuristic_partition(g, args):

    nx_g = dgl.to_networkx(g)
    start_time = time.time()

    nodes = set(nx_g.nodes())
    N = nx_g.number_of_nodes()
    E = nx_g.number_of_edges()
    average_degree = sum(nx_g.degree(v) for v in nx_g.nodes) / len(nx_g.nodes)
    candidate_vertices = [v for v in nx_g.nodes if abs(nx_g.degree(v) - average_degree) < 0.3 * average_degree]
    first_seed_vertex = random.choice(candidate_vertices)
    print(nx_g.degree(first_seed_vertex))
    seed_vertex_set = set()
    seed_vertex_set.add(first_seed_vertex)
    expandable_vertex_set = vertex_neighbor(nx_g, first_seed_vertex)
    p0_vertex = expandable_vertex_set
    p0_vertex.update(seed_vertex_set)
    p0_edge_set = vertex_edge(nx_g, first_seed_vertex)
    limit = E * args.edgelimit


    t = args.T
    while len(p0_edge_set) < limit:
        max_score = float('-inf')
        new_seed_vertex = None
        if p0_vertex - seed_vertex_set != 0:
            for vertex in (p0_vertex - seed_vertex_set):
                difference = vertex_neighbor(nx_g, vertex) - p0_vertex
                num_neighbor = len(vertex_neighbor(nx_g, vertex))
                score = -len(difference) + t * num_neighbor
                if score > max_score:
                    max_score = score
                    new_seed_vertex = vertex
        else:
            new_seed_vertex = random.choice(list(nodes - p0_vertex))
        seed_vertex_set.add(new_seed_vertex)
        new_seed_edge = vertex_edge(nx_g, new_seed_vertex)
        p0_edge_set.update(new_seed_edge)
        p0_vertex.update(vertex_neighbor(nx_g, new_seed_vertex))
        t = args.delta * t


    p1_edge_set = set(nx_g.edges()) - p0_edge_set
    p1_vertex = set()
    for edge in p1_edge_set:
        p1_vertex.add(edge[0])
        p1_vertex.add(edge[1])
    end_time = time.time()
    part_time = end_time - start_time
    print("partition time:{0:.3f}s".format(part_time))

    train_idx = (g.ndata["train_mask"] == True).nonzero(as_tuple=True)[0]
    train_nid = set(train_idx.tolist())

    src0, dst0 = zip(*p0_edge_set)
    src1, dst1 = zip(*p1_edge_set)

    eid0 = g.edge_ids(src0, dst0)

    eid1 = g.edge_ids(src1, dst1)
    subgraph0 = dgl.edge_subgraph(g, eid0)
    subgraph1 = dgl.edge_subgraph(g, eid1)

    p0_degree = subgraph0.in_degrees().float()
    p0_average_degree = p0_degree.mean().item()

    p0_gini = gini_torch(p0_degree)
    p0_Hen = edge_nor_entroy(p0_degree)
    p0_N = subgraph0.number_of_nodes()
    p0_E = subgraph0.number_of_edges()
    p0_complexity = p0_gini * p0_Hen * (p0_average_degree) ** args.gamma
    print("avg degrees of subgraph0：{0}".format(p0_average_degree))
    print("gini of subgraph0：{0}".format(p0_gini))
    print("Hen of subgraph0：{0}".format(p0_Hen))
    print("complexity of subgraph0".format(p0_complexity))
    mutildi_p0_edge_set = [(x, y, 0) for x, y in p0_edge_set]
    nx_sg0 = nx_g.edge_subgraph(mutildi_p0_edge_set)

    p1_degree = subgraph1.in_degrees().float()
    p1_average_degree = p1_degree.mean().item()

    p1_gini = gini_torch(p1_degree)
    p1_Hen = edge_nor_entroy(p1_degree)
    p1_N = subgraph0.number_of_nodes()
    p1_E = subgraph0.number_of_edges()
    p1_complexity = p1_gini * p1_Hen * (p1_average_degree) ** args.gamma
    print("avg degrees of subgraph1：{0}".format(p1_average_degree))
    print("gini of subgraph1：{0}".format(p1_gini))
    print("Hen of subgraph1：{0}".format(p1_Hen))
    print("complexity of subgraph0".format(p1_complexity))
    mutildi_p1_edge_set = [(x, y, 0) for x, y in p1_edge_set]
    nx_sg1 = nx_g.edge_subgraph(mutildi_p1_edge_set)

    cut_vertices = p0_vertex & p1_vertex
    cut_train_idx = cut_vertices & train_nid
    p0_train_idx = p0_vertex & train_nid
    p1_train_idx = p1_vertex & train_nid
    local_p0_train = p0_train_idx - cut_train_idx
    local_p1_train = p1_train_idx - cut_train_idx
    print("cut train nodes:{}".format(len(cut_train_idx)))
    print("p0 local train nodes:{}".format(len(local_p0_train)))
    print("p1 local train nodes:{}".format(len(local_p1_train)))

    non_p0_train = set()
    new_p0_train = set()
    new_p1_train = set()
    non_p1_train = set()
    for cut_train_nid in cut_train_idx:
        score_p0 = -len(vertex_neighbor(nx_g, cut_train_nid) - vertex_neighbor(nx_sg0, cut_train_nid)) * args.alpha - abs(
                    ((p0_complexity / p1_complexity) * (len(local_p0_train) + 1) / len(local_p1_train)) - args.m) * args.beta
        score_p1 = -len(vertex_neighbor(nx_g, cut_train_nid) - vertex_neighbor(nx_sg1, cut_train_nid)) * args.alpha - abs(
                    ((p1_complexity / p0_complexity) * (len(local_p1_train) + 1) / len(local_p0_train)) - 1 / args.m) * args.beta

        if score_p0 > score_p1:
            p1_train_idx.remove(cut_train_nid)
            non_p1_train.add(cut_train_nid)
            new_p0_train.add(cut_train_nid)
            local_p0_train.add(cut_train_nid)

        else:
            p0_train_idx.remove(cut_train_nid)
            non_p0_train.add(cut_train_nid)
            new_p1_train.add(cut_train_nid)
            local_p1_train.add(cut_train_nid)
    num_cut = len(cut_vertices)
    print("num of cut_vertices：", num_cut)

    new_p0_edge = p0_edge_set
    new_p1_edge = p1_edge_set

    for cut_v in cut_vertices:
        if len(vertex_neighbor(nx_g, cut_v) - vertex_neighbor(nx_sg0, cut_v)) > len(vertex_neighbor(nx_g, cut_v)) * args.lambda1:
            cut_v_edge = vertex_edge(nx_g, cut_v)
            new_p0_edge = new_p0_edge.union(cut_v_edge)
        elif len(vertex_neighbor(nx_g, cut_v) - vertex_neighbor(nx_sg1, cut_v)) > len(vertex_neighbor(nx_g, cut_v)) * args.lambda1:
            cut_v_edge = vertex_edge(nx_g, cut_v)
            new_p1_edge = new_p1_edge.union(cut_v_edge)
        else:
            continue

    cache_p0_edge = new_p0_edge - p0_edge_set
    cache_p0_vertex = set()
    for cache_e in cache_p0_edge:
        cache_p0_vertex.add(cache_e[0])
        cache_p0_vertex.add(cache_e[1])
    cache_p0_vertex = cache_p0_vertex - p0_vertex

    cache_p1_edge = new_p1_edge - p1_edge_set
    cache_p1_vertex = set()
    for cache_e in cache_p1_edge:
        cache_p1_vertex.add(cache_e[0])
        cache_p1_vertex.add(cache_e[1])
    cache_p1_vertex = cache_p1_vertex - p1_vertex

    src0, dst0 = zip(*new_p0_edge)
    src1, dst1 = zip(*new_p1_edge)
    new_eid0 = g.edge_ids(src0, dst0)

    new_eid1 = g.edge_ids(src1, dst1)
    new_subgraph0 = dgl.edge_subgraph(g, new_eid0)

    p0_nodes = new_subgraph0.number_of_nodes()
    p0_inner_node = torch.ones(p0_nodes, dtype=torch.int32)
    new_subgraph0.ndata["inner_node"] = p0_inner_node
    for cache_node in cache_p0_vertex:
        cache_nid = (new_subgraph0.ndata["_ID"] == cache_node).nonzero().item()
        new_subgraph0.ndata["inner_node"][cache_nid] = torch.tensor([0], dtype=torch.int32)
    new_subgraph0.ndata['train_mask'][(new_subgraph0.ndata['inner_node'] == 0).nonzero(as_tuple=True)[0]] = False
    for non_train_node in non_p0_train:
        non_train_nid = (new_subgraph0.ndata["_ID"] == non_train_node).nonzero().item()
        new_subgraph0.ndata["train_mask"][non_train_nid] = torch.tensor([False], dtype=torch.bool)
    new_subgraph0.ndata['train_mask'][(new_subgraph0.ndata['inner_node'] == 0).nonzero(as_tuple=True)[0]] = False
    new_subgraph0.ndata['val_mask'][(new_subgraph0.ndata['inner_node'] == 0).nonzero(as_tuple=True)[0]] = False
    new_subgraph0.ndata['test_mask'][(new_subgraph0.ndata['inner_node'] == 0).nonzero(as_tuple=True)[0]] = False

    new_subgraph1 = dgl.edge_subgraph(g, new_eid1)

    p1_nodes = new_subgraph1.number_of_nodes()
    p1_inner_node = torch.ones(p1_nodes, dtype=torch.int32)
    new_subgraph1.ndata["inner_node"] = p1_inner_node
    for cache_node in cache_p1_vertex:
        cache_nid = (new_subgraph1.ndata["_ID"] == cache_node).nonzero().item()
        new_subgraph1.ndata["inner_node"][cache_nid] = torch.tensor([0], dtype=torch.int32)
    new_subgraph1.ndata['train_mask'][(new_subgraph1.ndata['inner_node'] == 0).nonzero(as_tuple=True)[0]] = False
    for non_train_node in non_p1_train:
        non_train_nid = (new_subgraph1.ndata["_ID"] == non_train_node).nonzero().item()
        new_subgraph1.ndata["train_mask"][non_train_nid] = torch.tensor([False], dtype=torch.bool)
    new_subgraph1.ndata['train_mask'][(new_subgraph1.ndata['inner_node'] == 0).nonzero(as_tuple=True)[0]] = False
    new_subgraph1.ndata['val_mask'][(new_subgraph1.ndata['inner_node'] == 0).nonzero(as_tuple=True)[0]] = False
    new_subgraph1.ndata['test_mask'][(new_subgraph1.ndata['inner_node'] == 0).nonzero(as_tuple=True)[0]] = False
    print(new_subgraph0)
    print(new_subgraph1)
    print(len((new_subgraph0.ndata["train_mask"] == True).nonzero(as_tuple=True)[0]))
    print(len((new_subgraph1.ndata["train_mask"] == True).nonzero(as_tuple=True)[0]))
    print(len((g.ndata["train_mask"] == True).nonzero(as_tuple=True)[0]))
    dgl.save_graphs("CAGP_part/{}_sg0.bin".format(args.dataset), [new_subgraph0])
    dgl.save_graphs("CAGP_part/{}_sg1.bin".format(args.dataset), [new_subgraph1])


def main():
    args = perpare()
    if args.dataset == "cora":
        dataset = dgl.load_graphs("self_graph/cora.bin")
    elif args.dataset == "Pubmed":
        dataset = dgl.load_graphs("self_graph/Pubmed.bin")
    elif args.dataset == "Flickr":
        dataset = dgl.load_graphs("self_graph/Flickr.bin")
    elif args.dataset == "cophy":
        dataset = dgl.load_graphs("self_graph/cophy.bin")
    elif args.dataset == "corafull":
        dataset = dgl.load_graphs("self_graph/corafull.bin")
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    g = dataset[0][0]
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    print(args.dataset)
    print(g)
    heuristic_partition(g, args)

if __name__ == "__main__":
    main()