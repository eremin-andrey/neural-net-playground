"""Обучающая программа (без графики).

Запуск:  python3 train.py [--func mul] [--domain 01] [--epochs 1500] [--lr 0.5] [--hidden 8]
Функции: mul (x1×x2), add (x1+x2), div (деление), xor (XOR-ступенька), absdiff (|x1−x2|),
         pow (x1^x2, только для входов 0..1), exp (e^x2),
         hat (сомбреро), egg (яичная коробка), ring (кольцо).
Входы (--domain): 01 — от 0 до 1, pm1 — от −1 до +1.
--r2 — подсказка: третий вход r² = x1² + x2² (сильно помогает кольцу и сомбреро).
Результат: веса сохраняются в weights.json (для mul на 0..1) или weights_<func>[_pm1][_r2].json,
их можно загрузить кнопкой «Загрузить веса» в web_app.py.
"""
import argparse
import random
import time

from network import (DOMAINS, FUNCTIONS, NeuralNetwork, formula, has_domain, make_dataset,
                     target_function, weights_file)


def main():
    p = argparse.ArgumentParser(description="Обучение сети функции двух чисел")
    p.add_argument("--func", choices=FUNCTIONS, default="mul", help="чему учить сеть")
    p.add_argument("--domain", choices=DOMAINS, default="01", help="диапазон входов: 01 или pm1 (−1..+1)")
    p.add_argument("--epochs", type=int, default=1500, help="число эпох")
    p.add_argument("--lr", type=float, default=0.5, help="скорость обучения")
    p.add_argument("--hidden", type=int, default=8, help="нейронов в скрытом слое")
    p.add_argument("--samples", type=int, default=200, help="обучающих примеров")
    p.add_argument("--seed", type=int, default=1, help="зерно случайности")
    p.add_argument("--r2", action="store_true", help="подсказка: третий вход r² = x1² + x2²")
    p.add_argument("--out", help="файл для весов (по умолчанию weights.json / weights_<func>[_pm1][_r2].json)")
    args = p.parse_args()
    if not has_domain(args.func, args.domain):
        p.error(f"функция {args.func} не определена для входов {DOMAINS[args.domain][0]}")
    out = args.out or weights_file(args.func, args.domain, args.r2)
    _, lo, hi = DOMAINS[args.domain]

    train_data = make_dataset(args.samples, seed=args.seed, func=args.func, domain=args.domain)
    test_data = make_dataset(100, seed=args.seed + 1000, func=args.func, domain=args.domain)
    net = NeuralNetwork(n_hidden=args.hidden, seed=args.seed, r2_input=args.r2)
    rnd = random.Random(args.seed)

    print(f"Функция: {formula(args.func, args.domain)}, входы {DOMAINS[args.domain][0]}")
    print(f"Сеть {net.n_inputs}-{args.hidden}-1{' (подсказка r²)' if args.r2 else ''}, {args.samples} примеров, lr={args.lr}")
    print(f"{'эпоха':>7} {'MSE обуч.':>12} {'MSE пров.':>12}")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        mse = net.train_epoch(train_data, args.lr, rnd)
        if epoch == 1 or epoch % max(1, args.epochs // 15) == 0 or epoch == args.epochs:
            print(f"{epoch:>7} {mse:>12.6f} {net.evaluate(test_data):>12.6f}")
    print(f"Обучение заняло {time.time() - start:.1f} с")

    print("\nПроверка:")
    print(f"{'x1':>6} {'x2':>6} {'точно':>8} {'сеть':>8} {'ошибка':>8}")
    for u1, u2 in [(0, 0), (1, 1), (0.5, 0.5), (0.2, 0.9), (0.3, 0.7), (1, 0), (0.8, 0.6)]:
        x1, x2 = lo + (hi - lo) * u1, lo + (hi - lo) * u2  # точки проверки внутри диапазона входов
        t, y = target_function(x1, x2, args.func, args.domain), net.predict(x1, x2)
        print(f"{x1:>6.2f} {x2:>6.2f} {t:>8.4f} {y:>8.4f} {abs(y - t):>8.4f}")

    net.save(out, args.func, args.domain)
    print(f"\nВеса сохранены в {out}")


if __name__ == "__main__":
    main()
