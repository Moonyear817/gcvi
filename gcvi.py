"""GCVI：持续经营脆弱性指数的精确有理数实现。"""

import argparse
from fractions import Fraction as Fr
from itertools import product
import json
import math

GRADES = (0, 25, 50, 75, 95)
UPPER_B = (0, 49, 74, 94, 99)
CVALS = (0, 5, 10, 15)
N = 6
ALPHA = Fr(1, 10)
CAP = 99


def admissible(r, c):
    return c == 0 or sum(x > 0 for x in r) >= 2


def rhu(value):
    return math.floor(value + Fr(1, 2))


def a_raw(r, c, alpha=ALPHA, cap=CAP):
    maximum = max(r)
    return min(Fr(cap), maximum + alpha * (sum(r) - maximum) + c)


def b_raw(r, c, alpha=ALPHA, cmax=15):
    maximum = max(r)
    if maximum == 0:
        return Fr(0)
    upper = UPPER_B[GRADES.index(maximum)]
    stacked = alpha * (sum(r) - maximum) + c
    budget = alpha * (N - 1) * maximum + cmax
    return maximum + (upper - maximum) * stacked / budget


def evaluate(r, c=0, scheme="B", fatal="unknown", reviewed=True):
    """None 表示未知，c=None 枚举全部作用链分；区间只跨可行补全。"""
    if len(r) != N or any(x is not None and (type(x) is not int or x not in GRADES) for x in r):
        raise ValueError("必须提供六个档位，每项为 0、25、50、75、95 或 None")
    if c is not None and (type(c) is not int or c not in CVALS):
        raise ValueError("作用链分必须为 0、5、10、15 或 None")
    if scheme not in ("A", "B") or fatal not in ("true", "false", "unknown"):
        raise ValueError("方案或致命条件状态无效")
    result = {"scheme": scheme, "grades": list(r), "chain_score": c, "fatal": fatal}
    if fatal == "true":
        return dict(result, status="fatal", score=100)
    if fatal == "unknown" or not reviewed:
        return dict(result, status="pending", reason="致命状态未知或材料尚未复核")
    function = a_raw if scheme == "A" else b_raw
    scores = []
    for state in product(*(GRADES if x is None else (x,) for x in r)):
        for chain in CVALS if c is None else (c,):
            if admissible(state, chain):
                scores.append(function(state, chain))
    if not scores:
        raise ValueError("不存在可行状态：作用链加分要求至少两个系统存在非零风险")
    low, high = min(scores), max(scores)
    if None in r or c is None:
        return dict(result, status="interval", lower=rhu(low), upper=rhu(high),
                    raw_lower=str(low), raw_upper=str(high), completions=len(scores))
    return dict(result, status="point", score=rhu(low), raw=str(low))


def main():
    parser = argparse.ArgumentParser(description="GCVI 计算实现：输入经人工复核的六维状态编码。")
    parser.add_argument("grades", nargs=6, choices=[str(x) for x in GRADES] + ["?"],
                        help="现金、业务与收入、交付、团队、治理、必要经营条件；? 表示未知")
    parser.add_argument("--chains", default="0", choices=[str(x) for x in CVALS] + ["?"])
    parser.add_argument("--scheme", default="B", choices=["A", "B"])
    parser.add_argument("--fatal", default="unknown", choices=["true", "false", "unknown"])
    parser.add_argument("--unreviewed", action="store_true")
    args = parser.parse_args()
    try:
        result = evaluate([None if x == "?" else int(x) for x in args.grades],
                          None if args.chains == "?" else int(args.chains),
                          args.scheme, args.fatal, not args.unreviewed)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
