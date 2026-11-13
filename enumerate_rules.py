"""GCVI：方案A与方案B在有限形式域上的枚举校核。

本脚本只检验规则的数学性质，并在形式状态集合上计算描述统计。
状态集合上的均匀计数不是真实企业的分布，所得比例不是经验频率。

运行：python3 enumerate_rules.py（完整两两比较需要 numpy；输出 results.json）
"""
from fractions import Fraction as Fr
from itertools import product
import json
import math
import random
import time


from gcvi import GRADES, UPPER_B, CVALS, N, ALPHA, CAP, admissible, rhu, a_raw, b_raw


def build_states():
    states = [(r, c) for r in product(GRADES, repeat=N) for c in CVALS if admissible(r, c)]
    return states


def check_basic(states):
    out = {}
    out["n_states"] = len(states)
    out["n_vectors"] = len(GRADES) ** N
    # 最高单项下界与单一风险保真
    out["A_lower_bound_M"] = all(a_raw(r, c) >= max(r) for r, c in states)
    out["B_lower_bound_M"] = all(b_raw(r, c) >= max(r) for r, c in states)
    fid = True
    for u in GRADES:
        for i in range(N):
            r = tuple(u if j == i else 0 for j in range(N))
            fid &= a_raw(r, 0) == u and b_raw(r, 0) == u
    out["single_risk_fidelity_A_and_B"] = fid
    out["A_bounded_99"] = all(a_raw(r, c) <= 99 for r, c in states)
    out["B_bounded_99"] = all(b_raw(r, c) <= 99 for r, c in states)
    return out


def up_neighbors(r, c):
    """支配序上的单步上移：某一项升一档，或 C 加 5（保持可行）。"""
    for i in range(N):
        k = GRADES.index(r[i])
        if k < len(GRADES) - 1:
            r2 = r[:i] + (GRADES[k + 1],) + r[i + 1:]
            if admissible(r2, c):
                yield r2, c
    if c < 15 and admissible(r, c + 5):
        yield r, c + 5


def check_monotonicity(states):
    res = {"A_raw": True, "A_disp": True, "B_raw": True, "B_disp": True,
           "A_strict_nonmax_uncapped": True, "B_strict_nonmax_same_tier": True}
    for r, c in states:
        a0, b0 = a_raw(r, c), b_raw(r, c)
        for r2, c2 in up_neighbors(r, c):
            a1, b1 = a_raw(r2, c2), b_raw(r2, c2)
            res["A_raw"] &= a1 >= a0
            res["A_disp"] &= rhu(a1) >= rhu(a0)
            res["B_raw"] &= b1 >= b0
            res["B_disp"] &= rhu(b1) >= rhu(b0)
            # 严格叠加：最高档不变且有风险（M>0）时
            if max(r2) == max(r) and max(r) > 0 and (r2, c2) != (r, c):
                res["B_strict_nonmax_same_tier"] &= b1 > b0
                uncapped = max(r2) + ALPHA * (sum(r2) - max(r2)) + c2 <= CAP
                if uncapped:
                    res["A_strict_nonmax_uncapped"] &= a1 > a0
    return res


def tier_ranges(states):
    out = {}
    for name, f in (("A", a_raw), ("B", b_raw)):
        d = {}
        for r, c in states:
            m = max(r)
            v = f(r, c)
            lo, hi = d.get(m, (v, v))
            d[m] = (min(lo, v), max(hi, v))
        out[name] = {str(m): [str(lo), str(hi), rhu(lo), rhu(hi)] for m, (lo, hi) in sorted(d.items())}
    return out


def escalation_values():
    out = {}
    for name, f in (("A", a_raw), ("B", b_raw)):
        out[name] = {}
        for k in (1, 2, 3):
            v = f((GRADES[k],) * N, 15)
            out[name][str(GRADES[k])] = [str(v), float(v), v >= GRADES[k + 1]]
    return out


def a_family_region():
    """A族 min(99, M+αS+C) 中 C∈{0,c/3,2c/3,c}：枚举 (M,S) 可达组合，
    检验 '一档优先' 与 '两档优先' 的参数区域是否与解析条件一致（原始值）。"""
    ms_pairs = set()
    for r in product(GRADES, repeat=N):
        m = max(r)
        ms_pairs.add((m, sum(r) - m, sum(1 for x in r if x > 0)))
    mismatches_1, mismatches_2, tested = 0, 0, 0
    for ia in range(0, 201):               # α = 0.000, 0.001, ..., 0.200
        alpha = Fr(ia, 1000)
        for ic in range(0, 61):            # c = 0, 0.5, ..., 30
            cmax = Fr(ic, 2)
            hi = {}
            lo = {}
            for m, s, nnz in ms_pairs:
                cs = [Fr(0)] if nnz < 2 else [Fr(0), cmax / 3, 2 * cmax / 3, cmax]
                for cc in cs:
                    v = min(Fr(CAP), m + alpha * s + cc)
                    hi[m] = max(hi.get(m, v), v)
                    lo[m] = min(lo.get(m, v), v)
            ks = sorted(hi)
            p1 = all(hi[ks[i]] < lo[ks[i + 1]] for i in range(len(ks) - 1))
            p2 = all(hi[ks[i]] < lo[ks[i + 2]] for i in range(len(ks) - 2))
            pred1 = 375 * alpha + cmax < 20
            pred2 = 250 * alpha + cmax < 45
            mismatches_1 += p1 != pred1
            mismatches_2 += p2 != pred2
            tested += 1
    return {"param_points": tested, "one_tier_mismatch": mismatches_1, "two_tier_mismatch": mismatches_2}


def pairwise_statistics(states):
    import numpy as np
    M = np.array([max(r) for r, _ in states])
    A = np.array([float(a_raw(r, c)) for r, c in states])
    B = np.array([float(b_raw(r, c)) for r, c in states])
    Ad = np.array([rhu(a_raw(r, c)) for r, c in states])
    Bd = np.array([rhu(b_raw(r, c)) for r, c in states])
    R = np.array([r for r, _ in states], dtype=np.int16)
    C = np.array([c for _, c in states], dtype=np.int16)
    n = len(states)
    total_pairs = n * (n - 1) // 2

    out = {"distinct_display_A": int(len(set(Ad.tolist()))),
           "distinct_display_B": int(len(set(Bd.tolist()))),
           "distinct_raw_A": int(len(set(A.tolist()))),
           "distinct_raw_B": int(len(set(B.tolist())))}

    tie_A = tie_B = 0
    diffM = invA = tieA_diffM = invB = 0
    conc = disc = tie_x = tie_y = 0
    lipA = lipB = 0.0
    boundA_viol = 0
    worstB = None
    t0 = time.time()
    for i in range(n - 1):
        j = slice(i + 1, n)
        dM = M[j] - M[i]
        dA, dB = Ad[j] - Ad[i], Bd[j] - Bd[i]
        tie_A += int(np.sum(dA == 0))
        tie_B += int(np.sum(dB == 0))
        nz = dM != 0
        diffM += int(np.sum(nz))
        sgn = np.sign(dM[nz])
        invA += int(np.sum(np.sign(dA[nz]) == -sgn))
        tieA_diffM += int(np.sum(dA[nz] == 0))
        invB += int(np.sum(np.sign(dB[nz]) != sgn))
        # Kendall tau-b（A显示分 vs B原始分）
        sa, sb = np.sign(Ad[j] - Ad[i]), np.sign(B[j] - B[i])
        prod_ = sa * sb
        conc += int(np.sum(prod_ > 0))
        disc += int(np.sum(prod_ < 0))
        tie_x += int(np.sum(sa == 0))
        tie_y += int(np.sum(sb == 0))
        # Lipschitz 比率 |Δg| / (||Δr||∞ + |ΔC|)
        d = np.max(np.abs(R[j] - R[i]), axis=1)
        dc = np.abs(C[j] - C[i])
        den = d + dc
        ok = den > 0
        rA = np.abs(A[j] - A[i])[ok] / den[ok]
        rB = np.abs(B[j] - B[i])[ok] / den[ok]
        lipA = max(lipA, float(rA.max()) if rA.size else 0.0)
        if rB.size and rB.max() > lipB:
            lipB = float(rB.max())
            k = int(np.argmax(np.where(ok, np.abs(B[j] - B[i]) / np.where(den > 0, den, 1), -1)))
            worstB = [list(map(int, R[i])), int(C[i]), list(map(int, R[i + 1 + k])), int(C[i + 1 + k])]
        boundA_viol += int(np.sum(np.abs(A[j] - A[i]) > 1.5 * d + dc + 1e-9))
    n0 = total_pairs - tie_x
    n1 = total_pairs - tie_y
    tau_b = (conc - disc) / math.sqrt(n0 * n1)
    out.update({
        "total_pairs": total_pairs,
        "tie_share_display_A": tie_A / total_pairs,
        "tie_share_display_B": tie_B / total_pairs,
        "pairs_with_different_M": diffM,
        "A_strict_inversions_vs_M": invA,
        "A_strict_inversion_share": invA / diffM,
        "A_ties_with_different_M": tieA_diffM,
        "A_tie_share_with_different_M": tieA_diffM / diffM,
        "B_display_violations_of_M_order": invB,
        "kendall_tau_b_Adisplay_vs_Braw": tau_b,
        "lipschitz_ratio_A_raw": lipA,
        "lipschitz_ratio_B_raw": lipB,
        "lipschitz_worst_pair_B": worstB,
        "A_bound_1.5d_plus_dC_violations": boundA_viol,
        "seconds": time.time() - t0,
    })
    return out


def kleene_exactness(trials=20000, atoms=5, seed=7):
    """检验：不含否定的 AND/OR 公式，Kleene 三值求值 = 所有补全取值一致时的值，否则为 U。"""
    U = None
    rng = random.Random(seed)

    def rand_formula(depth):
        if depth == 0 or rng.random() < 0.3:
            return ("v", rng.randrange(atoms))
        return (rng.choice(("and", "or")), [rand_formula(depth - 1) for _ in range(rng.randint(2, 3))])

    def ev3(f, a):
        if f[0] == "v":
            return a[f[1]]
        vals = [ev3(g, a) for g in f[1]]
        if f[0] == "or":
            if 1 in vals:
                return 1
            return 0 if all(v == 0 for v in vals) else U
        if 0 in vals:
            return 0
        return 1 if all(v == 1 for v in vals) else U

    mismatch = 0
    for _ in range(trials):
        f = rand_formula(4)
        a = [rng.choice((0, 1, U)) for _ in range(atoms)]
        unk = [i for i, v in enumerate(a) if v is U]
        comps = set()
        for bits in product((0, 1), repeat=len(unk)):
            b = list(a)
            for i, x in zip(unk, bits):
                b[i] = x
            comps.add(ev3(f, b))
        truth = comps.pop() if len(comps) == 1 else U
        mismatch += ev3(f, a) != truth
    return {"trials": trials, "mismatch": mismatch}


def worked_examples():
    ex = {}
    cases = {
        "case1": ((75, 50, 50, 25, 25, 0), 5),
        "case1_repaired": ((25, 50, 25, 25, 25, 0), 0),
        "case3_cap": ((95, 95, 75, 75, 50, 25), 15),
        "counterexample_all75": ((75,) * 6, 0),
        "counterexample_single95": ((95, 0, 0, 0, 0, 0), 0),
        "nonunique_alpha": ((50, 25, 0, 0, 0, 0), 0),
    }
    for k, (r, c) in cases.items():
        ex[k] = {"A_raw": str(a_raw(r, c)), "A_disp": rhu(a_raw(r, c)),
                 "B_raw": str(b_raw(r, c)), "B_raw_float": float(b_raw(r, c)), "B_disp": rhu(b_raw(r, c))}
    ex["nonunique_alpha"]["A_raw_alpha_0.05"] = str(a_raw((50, 25, 0, 0, 0, 0), 0, alpha=Fr(1, 20)))
    # 未知区间示例：现金75已核实，业务与收入未知，其余 (50,25,25,0)，已核实 C=5，额外作用链未知
    lo_r, hi_r = (75, 0, 50, 25, 25, 0), (75, 95, 50, 25, 25, 0)
    ex["interval_example"] = {
        "A_low": str(a_raw(lo_r, 5)), "A_high": str(a_raw(hi_r, 15)),
        "B_low": float(b_raw(lo_r, 5)), "B_high": float(b_raw(hi_r, 15)),
        "B_low_exact": str(b_raw(lo_r, 5)), "B_high_exact": str(b_raw(hi_r, 15)),
    }
    return ex


def main():
    states = build_states()
    results = {
        "note": "形式状态集合上的规则校核；不是企业经验数据。",
        "basic": check_basic(states),
        "monotonicity_single_step": check_monotonicity(states),
        "tier_ranges": tier_ranges(states),
        "escalation_all_same_grade_C15": escalation_values(),
        "a_family_parameter_region": a_family_region(),
        "kleene_positive_formula_exactness": kleene_exactness(),
        "examples": worked_examples(),
    }
    print(json.dumps(results, ensure_ascii=False, indent=1))
    results["pairwise"] = pairwise_statistics(states)
    print(json.dumps(results["pairwise"], ensure_ascii=False, indent=1))
    with open("results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
