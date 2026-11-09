"""Завдання 1: жадібний і динамічний алгоритми видачі решти.

Запуск: python3 task_1.py
Benchmark: python3 task_1.py --benchmark
"""

from __future__ import annotations

import argparse
from statistics import median
from timeit import Timer

DENOMINATIONS: tuple[int, ...] = (50, 25, 10, 5, 2, 1)


def _validate_amount(amount: int) -> None:
    """Кількість грошових одиниць має бути невід'ємним цілим числом."""
    if isinstance(amount, bool) or not isinstance(amount, int):
        raise TypeError("Сума має бути цілим числом (int).")
    if amount < 0:
        raise ValueError("Сума не може бути від'ємною.")


def find_coins_greedy(amount: int) -> dict[int, int]:
    """Видати решту жадібно: від найбільшого номіналу до найменшого.

    Повертає лише монети з ненульовою кількістю. Для цього набору
    номіналів коректність жадібного вибору не слід переносити на довільні
    системи номіналів без окремого доведення.
    """
    _validate_amount(amount)
    result: dict[int, int] = {}
    remaining = amount

    for coin in DENOMINATIONS:
        count, remaining = divmod(remaining, coin)
        if count:
            result[coin] = count

    return result


def find_min_coins(amount: int) -> dict[int, int]:
    """Знайти мінімальну кількість монет методом динамічного програмування.

    dp[value] зберігає мінімальну кількість монет для value.
    selected[value] зберігає номінал останньої вибраної монети.

    Час O(amount * len(DENOMINATIONS)), пам'ять O(amount).
    """
    _validate_amount(amount)
    if amount == 0:
        return {}

    unreachable = amount + 1
    dp = [0] + [unreachable] * amount
    selected = [0] * (amount + 1)

    for value in range(1, amount + 1):
        for coin in DENOMINATIONS:
            if coin <= value and dp[value - coin] + 1 < dp[value]:
                dp[value] = dp[value - coin] + 1
                selected[value] = coin

    if dp[amount] == unreachable:
        raise ValueError("Неможливо сформувати решту з доступних монет.")

    result: dict[int, int] = {}
    remaining = amount
    while remaining > 0:
        coin = selected[remaining]
        result[coin] = result.get(coin, 0) + 1
        remaining -= coin

    # Стабільний порядок для читабельного відображення.
    return {coin: result[coin] for coin in DENOMINATIONS if coin in result}


def coin_count(change: dict[int, int]) -> int:
    """Підрахувати загальну кількість монет."""
    return sum(change.values())


def benchmark(amounts: tuple[int, ...] = (113, 1_000, 10_000, 100_000)) -> None:
    """Вивести час виконання через timeit у мілісекундах на виклик.

    Кожний сценарій заміряється 7 разів. Кількість викликів за один
    замір відрізняється, але час нормалізується на один виклик.
    """
    print("\n## Порівняння продуктивності (timeit)")
    print("| Сума | Greedy, мс | DP, мс | Прискорення greedy | Оптимально greedy? |")
    print("|---:|---:|---:|---:|:---:|")
    for amount in amounts:
        greedy = Timer(lambda: find_coins_greedy(amount))
        dynamic = Timer(lambda: find_min_coins(amount))
        greedy_ms = median(greedy.repeat(repeat=7, number=1000)) / 1000 * 1_000
        dynamic_ms = median(dynamic.repeat(repeat=7, number=3)) / 3 * 1000
        greedy_count = coin_count(find_coins_greedy(amount))
        minimum_count = coin_count(find_min_coins(amount))
        optimal = "так" if greedy_count == minimum_count else "ні"
        print(
            f"| {amount:,} | {greedy_ms:.6f} | {dynamic_ms:.6f} | "
            f"{dynamic_ms / greedy_ms:.1f}x | {optimal} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Порівняння алгоритмів видачі решти")
    parser.add_argument("--amount", type=int, default=113, help="Сума для видачі решти")
    parser.add_argument("--benchmark", action="store_true", help="Запустити порівняння через timeit")
    args = parser.parse_args()

    for amount in dict.fromkeys((args.amount, 30)):
        greedy = find_coins_greedy(amount)
        optimal = find_min_coins(amount)
        print(f"Сума: {amount}")
        print(f"  Greedy: {greedy} (монет: {coin_count(greedy)})")
        print(f"  DP:     {optimal} (монет: {coin_count(optimal)})")

    if args.benchmark:
        benchmark()


if __name__ == "__main__":
    main()
