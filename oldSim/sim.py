import argparse
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, Counter

# ------------------------------------------
# LRU and LFU Baseline Cache Policies
# ------------------------------------------
class LRU_Policy:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = deque()
        self.set = set()
    def access(self, item):
        hit = item in self.set
        if hit:
            self.cache.remove(item)
            self.cache.appendleft(item)
        else:
            if len(self.cache) >= self.capacity:
                ev = self.cache.pop()
                self.set.remove(ev)
            self.cache.appendleft(item)
            self.set.add(item)
        return hit

class LFU_Policy:
    def __init__(self, capacity):
        self.capacity = capacity
        self.freq = Counter()
        self.set = set()
    def access(self, item):
        hit = item in self.set
        if not hit:
            if len(self.set) >= self.capacity:
                # evict least-freq
                ev = min(self.set, key=lambda x: self.freq[x])
                self.set.remove(ev)
                del self.freq[ev]
            self.set.add(item)
        self.freq[item] += 1
        return hit

# ------------------------------------------
# LEO Satellite Policy Simulation
# ------------------------------------------
class LEO_Satellite:
    def __init__(self, sat_id, lat, lon, Z, D, c_m, c_f, c_s, alpha, c_p_base, c_isl, hops, policy_type):
        self.id = sat_id
        self.lat = lat
        self.lon = lon
        self.Z = Z
        self.D = D
        self.c_m = c_m
        self.c_f = c_f
        self.c_s = c_s
        self.alpha = alpha
        self.c_p_base = c_p_base
        self.c_isl = c_isl
        self.hops = hops
        # select policy
        if policy_type == 'LRU':
            self.policy = LRU_Policy(Z)
        else:
            self.policy = LFU_Policy(Z)
    def in_coverage(self, user_pos, psi):
        tu, pu = user_pos
        ts = math.radians(self.lat); ps = math.radians(self.lon)
        cosg = math.sin(tu)*math.sin(ts)*math.cos(pu-ps) + math.cos(tu)*math.cos(ts)
        return math.acos(max(-1, min(1, cosg))) <= psi
    def access(self, view, neighbor_cache, psi):
        # returns hit_type, cost components
        # hit_type: 'HIT','ISL','SYN','MISS'
        if self.policy.access(view):
            return 'HIT', [], [view], []
        if view in neighbor_cache:
            return 'ISL', [], [view], [view]
        left, right = view-1, view+1
        if abs(right-left)<=self.D and self.policy.access(left) and self.policy.access(right):
            return 'SYN', [], [left, right], []
        return 'MISS', [view], [view], []

# ------------------------------------------
# Request Trace Generation
# ------------------------------------------
def generate_trace(V, T, seed=None):
    if seed: random.seed(seed)
    w = np.array([1/(i+1) for i in range(V)])
    w /= w.sum()
    return random.choices(range(1,V+1), weights=w, k=T)

# ------------------------------------------
# Main Comparative Simulation (per-slot metrics)
# ------------------------------------------
def simulate(V, T, leo_params, num_sats, psi):
    trace = generate_trace(V, T, seed=42)
    results = {}
    for policy in ['LRU','LFU']:
        # init sats
        sats = [LEO_Satellite(i, random.uniform(-90,90), random.uniform(-180,180), *leo_params, policy)
                for i in range(num_sats)]
        slot_costs = []
        for view in trace:
            user_pos = (random.random()*math.pi, random.random()*2*math.pi)
            cov = [s for s in sats if s.in_coverage(user_pos, psi)]
            if not cov:
                slot_costs.append(0)
                continue
            sat = cov[0]
            nb_cache = set().union(*(n.policy.set for n in cov[1:]))
            ht, v_f, v_s, v_isl = sat.access(view, nb_cache, psi)
            cost = ( (sat.c_m if ht=='MISS' else 0)
                   + len(v_f)*sat.c_f
                   + len(v_s)*sat.c_s
                   + len(v_isl)*sat.c_isl*sat.hops
                   + (sat.alpha*abs(v_s[1]-v_s[0])+sat.c_p_base if ht=='SYN' else 0) )
            slot_costs.append(cost)
        results[policy] = slot_costs
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--V', type=int, default=4096)
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--sats', type=int, default=20)
    args = parser.parse_args()
    
    # ------------------------------------------
    # Parameter Assignment
    # ------------------------------------------
    # Vector size V: total distinct views
    # T: number of requests (simulation length)
    # capacity: LEO cache capacity (also controls LRU/LFU capacity)
    # sats: number of satellites in the constellation
    # psi: coverage half-cone angle for each LEO
    # leo_settings unpacks to:
    #   Z        # cache capacity per LEO (same as capacity here)
    #   D        # DIBR synthesis range
    #   c_m      # miss penalty cost
    #   c_f      # fetch cost from ground station
    #   c_s      # transmit cost to user
    #   alpha    # DIBR distance weight
    #   c_p_base # fixed DIBR synthesis cost
    #   c_isl    # ISL per-hop cost
    #   hops     # number of inter-satellite hops considered

    # Assign simulation parameters here:
    leo_settings = (
        32,    # Z: cache capacity in view units
        3,     # D: max DIBR synthesis distance
        10,    # c_m: cache miss penalty
        3,     # c_f: fetch cost
        2,     # c_s: transmission cost
        0.3,   # alpha: DIBR distance weight
        5,     # c_p_base: fixed DIBR cost
        2,     # c_isl: ISL per-hop cost
        1      # hops: number of ISL hops
    )
    psi = math.radians(30)  # coverage half-cone angle (30° converted to radians)
    # === End of parameter assignment ===

    res=simulate(args.V,args.T,leo_settings,args.sats,psi)
    # plot
    plt.figure(figsize=(10,4))
    for pol,col in zip(['LRU','LFU'],['skyblue','orange']):
        plt.plot(np.cumsum(res[pol]),label=f'{pol} CumCost',color=col)
    plt.xlabel('Time slot'); plt.ylabel('Cumulative Cost')
    plt.legend(); plt.grid(True)
    plt.show()

    # costs, lru_misses, lfu_misses = run_simulations(
    #     args.V, args.T, args.capacity, leo_settings, args.sats, psi)

    # # Plotting per-slot metrics
    # fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # # 1. Per-slot LEO cost
    # axes[0].plot(costs, color='green', linewidth=0.75)
    # axes[0].set_ylabel('LEO Cost')
    # axes[0].grid(True)

    # # 2. Per-slot LRU miss (0/1)
    # axes[1].plot(lru_misses, color='skyblue', linewidth=0.75)
    # axes[1].set_ylabel('LRU Miss (0/1)')
    # axes[1].grid(True)

    # # 3. Per-slot LFU miss (0/1)
    # axes[2].plot(lfu_misses, color='orange', linewidth=0.75)
    # axes[2].set_ylabel('LFU Miss (0/1)')
    # axes[2].grid(True)

    # # 4. Cumulative cost comparison
    # c_m_baseline = leo_settings[2]
    # lru_costs = np.cumsum([c_m_baseline if m==1 else 0 for m in lru_misses])
    # lfu_costs = np.cumsum([c_m_baseline if m==1 else 0 for m in lfu_misses])
    # leo_costs = np.cumsum(costs)

    # axes[3].plot(leo_costs, label='LEO Cum Cost', color='green', linewidth=0.8)
    # axes[3].plot(lru_costs, label='LRU Cum Cost', color='skyblue', linewidth=0.8)
    # axes[3].plot(lfu_costs, label='LFU Cum Cost', color='orange', linewidth=0.8)
    # axes[3].set_ylabel('Cumulative Cost')
    # axes[3].set_xlabel('Time Slot')
    # axes[3].legend(loc='upper left')
    # axes[3].grid(True)

    # plt.tight_layout()
    # plt.show()
