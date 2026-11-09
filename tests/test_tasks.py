"""Перевірки обох завдань. Запуск: python3 -m unittest discover -s tests -v"""

import math
import unittest
from collections import deque

from task_1 import DENOMINATIONS, coin_count, find_coins_greedy, find_min_coins
from task_2 import exact_integral, f, monte_carlo_integral


def reference_min_count(amount: int) -> int:
    """Незалежний BFS-еталон для малих сум (кожна монета — один крок)."""
    if amount == 0:
        return 0
    queue = deque([(0, 0)])
    seen = {0}
    while queue:
        total, steps = queue.popleft()
        for coin in DENOMINATIONS:
            next_total = total + coin
            if next_total == amount:
                return steps + 1
            if next_total < amount and next_total not in seen:
                seen.add(next_total)
                queue.append((next_total, steps + 1))
    raise AssertionError("Еталонний пошук не знайшов розв'язку")


class TestCoinChange(unittest.TestCase):
    def test_given_example(self):
        expected = {50: 2, 10: 1, 2: 1, 1: 1}
        self.assertEqual(find_coins_greedy(113), expected)
        self.assertEqual(find_min_coins(113), expected)

    def test_zero_amount(self):
        self.assertEqual(find_coins_greedy(0), {})
        self.assertEqual(find_min_coins(0), {})

    def test_thirty_uses_two_coins(self):
        self.assertEqual(find_coins_greedy(30), {25: 1, 5: 1})
        self.assertEqual(find_min_coins(30), {25: 1, 5: 1})

    def test_equal_counts_in_checked_range(self):
        # Це емпірична перевірка, не математичне доведення.
        for amount in range(0, 501):
            with self.subTest(amount=amount):
                self.assertEqual(
                    coin_count(find_coins_greedy(amount)),
                    coin_count(find_min_coins(amount)),
                )

    def test_negative_amount_rejected(self):
        for function in (find_coins_greedy, find_min_coins):
            with self.subTest(function=function.__name__):
                with self.assertRaises(ValueError):
                    function(-1)

    def test_non_integer_amount_rejected(self):
        for function in (find_coins_greedy, find_min_coins):
            for amount in (True, 1.5, "113", None):
                with self.subTest(function=function.__name__, amount=amount):
                    with self.assertRaises(TypeError):
                        function(amount)

    def test_both_reconstruct_correct_amount(self):
        for amount in range(0, 201):
            for function in (find_coins_greedy, find_min_coins):
                with self.subTest(amount=amount, method=function.__name__):
                    result = function(amount)
                    self.assertEqual(sum(coin * quantity for coin, quantity in result.items()), amount)
                    self.assertTrue(all(coin in DENOMINATIONS for coin in result))
                    self.assertTrue(all(quantity > 0 for quantity in result.values()))

    def test_dynamic_programming_matches_independent_bfs(self):
        for amount in range(0, 121):
            with self.subTest(amount=amount):
                self.assertEqual(coin_count(find_min_coins(amount)), reference_min_count(amount))

    def test_greedy_is_not_better_than_optimal(self):
        for amount in range(0, 301):
            self.assertGreaterEqual(coin_count(find_coins_greedy(amount)), coin_count(find_min_coins(amount)))


class TestMonteCarlo(unittest.TestCase):
    def test_analytic_integral(self):
        self.assertAlmostEqual(exact_integral(0, 2), 8 / 3)
        self.assertAlmostEqual(exact_integral(1, 3), 26 / 3)

    def test_function(self):
        self.assertEqual(f(0), 0)
        self.assertEqual(f(2), 4)
        self.assertEqual(f(-3), 9)

    def test_constant_function_exact(self):
        result, error = monte_carlo_integral(lambda x: 3.0, 0, 2, 100)
        self.assertAlmostEqual(result, 6.0)
        self.assertAlmostEqual(error, 0.0)

    def test_deterministic_reproducibility(self):
        first = monte_carlo_integral(f, 0, 2, 10_000, seed=123)
        second = monte_carlo_integral(f, 0, 2, 10_000, seed=123)
        self.assertEqual(first, second)

    def test_monte_carlo_accuracy(self):
        result, standard_error = monte_carlo_integral(f, 0, 2, 100_000, seed=42)
        self.assertLess(abs(result - 8 / 3), 0.03)
        self.assertGreater(standard_error, 0)

    def test_linear_function(self):
        result, standard_error = monte_carlo_integral(lambda x: x, -1, 1, 100_000)
        self.assertLess(abs(result), 0.03)
        self.assertGreater(standard_error, 0)

    def test_invalid_samples(self):
        for samples in (0, 1, -1):
            with self.subTest(samples=samples):
                with self.assertRaises(ValueError):
                    monte_carlo_integral(f, 0, 2, samples)
        for samples in (True, 2.5, "10"):
            with self.subTest(samples=samples):
                with self.assertRaises(TypeError):
                    monte_carlo_integral(f, 0, 2, samples)

    def test_invalid_bounds(self):
        for a, b in ((2, 0), (1, 1), (math.nan, 2), (0, math.inf)):
            with self.subTest(a=a, b=b):
                with self.assertRaises(ValueError):
                    monte_carlo_integral(f, a, b, 10)

    def test_nonfinite_function_rejected(self):
        with self.assertRaises(ValueError):
            monte_carlo_integral(lambda x: math.nan, 0, 2, 10)


if __name__ == "__main__":
    unittest.main()
