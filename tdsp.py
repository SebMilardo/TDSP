import numpy as np
from Queue import PriorityQueue
import matplotlib
matplotlib.use("tkagg")


class Graph(object):
    def __init__(self, vertices, edges, weights):
        self.vertices = vertices
        self.edges = edges
        self.weights = weights


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

# Given v_s and v_e and a time window T, find the best time within T to depart from v_s and the path along which one
# can arrive at v_e with the minimum travel time

# Algorithm 1

# Input:
# Gt = time dependent graph
# vs = starting v
# ve = destination v
# T  = start time interval

# Output
# p_ = optimal vs-ve path
# t_ = optimal starting time


def w1_3(t):
    if 0 <= t < 5:
        return 5
    if t >= 10:
        return 25
    if 5 <= t < 10:
        return 4 * t - 15


def w1_2(t): return 10


def w2_3(t): return 10


def w2_4(t): return 25


def w3_4(t): return 10 if t >= 40 else 42 - 4.0 / 5.0 * t


weights = {(1, 3): w1_3, (2, 3): w2_3, (1, 2): w1_2, (2, 4): w2_4, (3, 4): w3_4}
Gt = Graph({1: 1, 2: 2, 3: 3, 4: 4}, [(1, 3), (2, 3), (1, 2), (2, 4), (3, 4)], weights)


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
        tmp = [Gt.weights[e](g[pair_k.v][tau[pair_k.v]]) for e in Gt.edges if e[1] == pair_i.v]
        delta = np.min(tmp) if len(tmp) > 0 else np.inf
        tau_i_first = np.max([t for t in T if g[pair_i.v][t] <= (g[pair_k.v][tau[pair_k.v]] + delta)])

        for e in [e for e in Gt.edges if e[0] == pair_i.v]:
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
        for e in [e for e in Gt.edges if e[1] == vj]:
            if g[e[0]][t_star] + Gt.weights[e](g[e[0]][t_star]) == g[vj][t_star]:
                vj = e[0]
                break
        p_star.append([e[0], e[1], g[e[1]][t_star]])
    return list(reversed(p_star))


def algorithm1(Gt, vs, ve, T):
    g = time_refinement(Gt, vs, ve, T)

    if sum(np.isinf(g[ve].values())) == 0:
        t_star = np.argmin([g[ve][t] - t for t in T])
        p_star = path_selector(Gt, g, vs, ve, t_star)
        return t_star, p_star
    else:
        return None


print algorithm1(Gt, 1, 4, range(0, 61))
