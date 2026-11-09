"""Завдання 2: інтеграл f(x)=x² на [0, 2] методом Монте-Карло.

Запуск: python3 task_2.py
Графік: python3 task_2.py --plot integral_plot.png
"""

from __future__ import annotations

import argparse
import math
import random
from collections.abc import Callable


def f(x: float) -> float:
    """Функція, яку інтегруємо."""
    return x**2


def exact_integral(a: float = 0.0, b: float = 2.0) -> float:
    """Аналітичний результат: [x³ / 3] від a до b."""
    return (b**3 - a**3) / 3


def monte_carlo_integral(
    func: Callable[[float], float],
    a: float,
    b: float,
    n_samples: int,
    seed: int = 42,
) -> tuple[float, float]:
    """Оцінити інтеграл за середнім значенням випадкових вибірок.

    X ~ Uniform(a, b), I ≈ (b-a) * mean(func(X)).
    Повертає (оцінка інтеграла, оцінка стандартної похибки).
    Застосовується онлайн-оцінка дисперсії Велфорда: пам'ять O(1).
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise TypeError("Кількість точок має бути цілим числом.")
    if n_samples < 2:
        raise ValueError("Для оцінки стандартної похибки потрібно щонайменше 2 точки.")
    if not (math.isfinite(a) and math.isfinite(b)) or a >= b:
        raise ValueError("Межі мають бути скінченними числами з a < b.")

    rng = random.Random(seed)
    width = b - a
    mean = 0.0
    m2 = 0.0

    for index in range(1, n_samples + 1):
        x = rng.uniform(a, b)
        y = float(func(x))
        if not math.isfinite(y):
            raise ValueError("Функція повернула нескінченне або NaN значення.")
        delta = y - mean
        mean += delta / index
        m2 += delta * (y - mean)

    estimate = width * mean
    sample_variance = m2 / (n_samples - 1)
    standard_error = width * math.sqrt(sample_variance / n_samples)
    return estimate, standard_error


def plot_integral(path: str = "integral_plot.png") -> None:
    """Зберегти графік функції та області інтегрування (потрібен matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Для графіка встановіть matplotlib: pip install matplotlib") from exc

    a, b = 0.0, 2.0
    x = [-0.5 + i * 3.0 / 499 for i in range(500)]
    shaded_x = [a + i * (b - a) / 299 for i in range(300)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, [f(point) for point in x], color="firebrick", linewidth=2, label=r"$f(x)=x^2$")
    ax.fill_between(
        shaded_x,
        [f(point) for point in shaded_x],
        color="slategray", alpha=0.35,
        label=r"$\int_0^2 x^2 dx = 8/3$",
    )
    ax.axvline(a, color="gray", linestyle="--")
    ax.axvline(b, color="gray", linestyle="--")
    ax.set(xlabel="x", ylabel="f(x)", title="Monte Carlo integration: f(x) = x² over [0, 2]")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Графік збережено: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Інтеграл методом Монте-Карло")
    parser.add_argument(
        "--samples", type=int, nargs="+", default=[1_000, 10_000, 100_000, 1_000_000],
        help="Кількості випадкових точок для порівняння",
    )
    parser.add_argument("--seed", type=int, default=42, help="Початкове значення RNG")
    parser.add_argument("--plot", metavar="PATH", help="Зберегти графік у PNG (потрібен matplotlib)")
    args = parser.parse_args()

    exact = exact_integral()
    print("Інтеграл ∫₀² x² dx = 8/3 ≈", f"{exact:.9f}")
    print("| N | Монте-Карло | Абсолютна похибка | Відносна похибка, % | Оцінка SE |")
    print("|---:|---:|---:|---:|---:|")

    for n in args.samples:
        estimate, standard_error = monte_carlo_integral(f, 0, 2, n, seed=args.seed)
        absolute_error = abs(estimate - exact)
        relative_error = absolute_error / abs(exact) * 100
        print(
            f"| {n:,} | {estimate:.9f} | {absolute_error:.9f} | "
            f"{relative_error:.4f} | {standard_error:.9f} |"
        )

    if args.plot:
        plot_integral(args.plot)


if __name__ == "__main__":
    main()
