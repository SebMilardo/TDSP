import random
import time
from Queue import PriorityQueue

import matplotlib
import numpy as np

matplotlib.use("tkagg")
import pandas as pd


class Graph(object):
    def __init__(self, vertices, edges, weights):
        self.vertices = vertices
        self.edges = edges
        self.weights = weights
        self.in_adj = list()
        self.out_adj = list()

        for v in vertices:
            self.in_adj.append(list())
            self.out_adj.append(list())

        for e in edges:
            self.in_adj[e[1]].append(e)
            self.out_adj[e[0]].append(e)


class Pair(object):
    def __init__(self, tau, g, v):
        self.tau = tau
        self.g = g
        self.v = v

    def __cmp__(self, other):
        return cmp(self.g[self.tau[self.v]], other.g[other.tau[other.v]])

    def __str__(self):
        return str((self.v, self.tau[self.v], self.g[self.tau[self.v]]))

    def __repr__(self):
        return str((self.v, self.tau[self.v], self.g[self.tau[self.v]]))


def time_refinement(Gt, vs, ve, T):
    ts = T[0]
    te = T[-1]
    g = dict()
    g[vs] = dict()
    for t in T:
        g[vs][t] = t
    tau = dict()
    tau[vs] = ts

    for v in Gt.vertices:
        if v != vs:
            g[v] = dict()
            for t in T:
                g[v][t] = np.inf
            tau[v] = ts

    Q_ = [Pair(tau, g[i], i) for i in Gt.vertices]
    Q = PriorityQueue()
    for p in Q_:
        Q.put(p)

    while len(Q.queue) >= 2:
        pair_i = Q.get()
        pair_k = Q.get()
        Q.put(pair_k)
        tmp = [Gt.weights[e](g[pair_k.v][tau[pair_k.v]]) for e in Gt.in_adj[pair_i.v]]
        delta = np.min(tmp) if len(tmp) > 0 else np.inf
        tau_i_first = np.max([t for t in T if g[pair_i.v][t] <= (g[pair_k.v][tau[pair_k.v]] + delta)])

        for e in Gt.out_adj[pair_i.v]:
            gj_first = dict()
            for t in range(pair_i.tau[pair_i.v], tau_i_first + 1):
                gj_first[t] = pair_i.g[t] + Gt.weights[e](pair_i.g[t])
                g[e[1]][t] = min(g[e[1]][t], gj_first[t])

            tmpQ = PriorityQueue()
            for p in Q.queue:
                tmpQ.put(p)
            Q = tmpQ

        tau[pair_i.v] = tau_i_first

        if tau[pair_i.v] >= te:
            if pair_i.v == ve:
                return g
        else:
            Q.put(Pair(tau, pair_i.g, pair_i.v))
    return g


def path_selector(Gt, g, vs, ve, t_star):
    vj = ve
    p_star = list()
    while vj != vs:
        for e in Gt.in_adj[vj]:
            if g[e[0]][t_star] + Gt.weights[e](g[e[0]][t_star]) == g[vj][t_star]:
                vj = e[0]
                break
        p_star.append([e[0], e[1], g[e[1]][t_star]])
    return list(reversed(p_star))


def algorithm(Gt, vs, ve, T):
    g = time_refinement(Gt, vs, ve, T)

    if sum(np.isinf(g[ve].values())) == 0:
        t_star = np.argmin([g[ve][t] - t for t in T])
        p_star = path_selector(Gt, g, vs, ve, t_star)
        return t_star, p_star
    else:
        return None


if __name__ == "__main__":
    # Given v_s and v_e and a time window T, find the best time within T to depart from v_s and the path along which one
    # can arrive at v_e with the minimum travel time

    # algorithm

    # Input:
    # Gt = time dependent graph
    # vs = starting vertex
    # ve = destination vertex
    # T  = start time interval

    # Output:
    # t_ = optimal starting time
    # p_ = optimal vs-ve path. [vi, vj, time to reach vj]

    # Weights:
    # the weight of an edge v(i,j) depends on time t and it is defined by the function wi_j

    def w0_2(t):
        if 0 <= t < 5:
            return 5
        if t >= 10:
            return 25
        if 5 <= t < 10:
            return 4 * t - 15


    def w0_1(t):
        return 10


    def w1_2(t):
        return 10


    def w1_3(t):
        return 25


    def w2_3(t):
        return 10 if t >= 40 else 42 - 4.0 / 5.0 * t


    start_time = time.time()

    T = range(0, 61)
    vertices = {0: 0, 1: 1, 2: 2, 3: 3}
    vs = 0
    ve = 3
    edges = [(0, 2), (1, 2), (0, 1), (1, 3), (2, 3)]
    weights = {(0, 2): w0_2, (1, 2): w1_2, (0, 1): w0_1, (1, 3): w1_3, (2, 3): w2_3}
    Gt = Graph(vertices, edges, weights)
    print algorithm(Gt, vs, ve, T)

    print time.time() - start_time

    res = []
    for ii in range(1, 100):
        T2 = range(0, ii)
        vertices2 = dict()
        for i in range(ii): vertices2[i] = i
        l = len(vertices2)
        edges2 = [(i, j) for i in range(l) for j in range(l) if random.random() < 0.5]
        weights2 = dict()
        all_w = [w0_2, w1_2, w0_1, w1_3, w2_3]
        for e in edges2:
            weights2[e] = all_w[random.randint(0, 4)]
        Gt = Graph(vertices2, edges2, weights2)

        start_time = time.time()
        p = algorithm(Gt, random.randint(0, ii - 1), random.randint(0, ii - 1), T2)
        end = time.time() - start_time
        item = [ii, end, len(vertices2), len(edges2), len(T2)]
        print (p, item)
        res.append(item)

    df = pd.DataFrame(res, columns=["i", "time_elapsed", "n", "m", "T"])
    df["time_elapsed"].plot()
