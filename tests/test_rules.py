from fractions import Fraction
import unittest

from gcvi import a_raw, b_raw, evaluate, rhu
from enumerate_rules import build_states, check_basic, check_monotonicity, tier_ranges


class RulesTest(unittest.TestCase):
    def test_published_examples(self):
        for risks, chain, a, b in [
            ((75, 50, 50, 25, 25, 0), 5, 95, 82),
            ((25, 50, 25, 25, 25, 0), 0, 60, 56),
            ((95, 95, 75, 75, 50, 25), 15, 99, 98),
            ((75,) * 6, 0, 99, 89),
            ((95, 0, 0, 0, 0, 0), 0, 95, 95),
        ]:
            with self.subTest(risks=risks):
                self.assertEqual(rhu(a_raw(risks, chain)), a)
                self.assertEqual(rhu(b_raw(risks, chain)), b)

    def test_half_up(self):
        self.assertEqual(rhu(Fraction(105, 2)), 53)

    def test_fatal_overrides_missing_information(self):
        self.assertEqual(evaluate([None] * 6, fatal="true", reviewed=False)["score"], 100)

    def test_unknown_and_unreviewed_do_not_get_point_score(self):
        self.assertEqual(evaluate([0] * 6)["status"], "pending")
        self.assertEqual(evaluate([0] * 6, fatal="false", reviewed=False)["status"], "pending")

    def test_partial_information(self):
        result = evaluate([75, None, 50, 25, 25, 0], None, fatal="false")
        self.assertEqual((result["lower"], result["upper"]), (79, 97))
        result = evaluate([75, None, 50, 25, 25, 0], 5, fatal="false")
        self.assertEqual((result["lower"], result["upper"]), (80, 96))

    def test_interval_filters_infeasible_completions(self):
        result = evaluate([None, None, 0, 0, 0, 0], 5, fatal="false")
        self.assertEqual(result["completions"], 16)

    def test_input_validation(self):
        for risks, chain in [([0] * 5, 0), ([12] * 6, 0), ([0] * 6, 5), ([True] * 6, 0)]:
            with self.subTest(risks=risks, chain=chain), self.assertRaises(ValueError):
                evaluate(risks, chain, fatal="false")

    def test_all_formal_states(self):
        states = build_states()
        self.assertEqual(len(states), 62425)
        basic = check_basic(states)
        self.assertTrue(all(value for key, value in basic.items() if key not in ("n_states", "n_vectors")))
        self.assertTrue(all(check_monotonicity(states).values()))
        ranges = tier_ranges(states)["B"]
        self.assertEqual([ranges[str(m)][2:] for m in (0, 25, 50, 75, 95)],
                         [[0, 0], [25, 49], [50, 74], [75, 94], [95, 99]])


if __name__ == "__main__":
    unittest.main()
