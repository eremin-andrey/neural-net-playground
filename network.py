"""Простая нейронная сеть на чистом Python (без numpy).

Архитектура: 2 входа (+ необязательный третий вход-подсказка r² = x1² + x2²)
-> скрытый слой (sigmoid) -> 1 выход (линейный).
Обучается методом обратного распространения ошибки (backpropagation)
и стохастическим градиентным спуском.
"""
import json
import math
import random


def sigmoid(x):
    # Защита от переполнения math.exp при больших по модулю x
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


# Диапазоны входов: x1 и x2 берутся из [от, до]
DOMAINS = {
    "01": ("0..1", 0.0, 1.0),
    "pm1": ("−1..+1", -1.0, 1.0),
}
DEFAULT_DOMAIN = "01"   # для имён файлов и старых файлов весов без пометки диапазона
START_DOMAIN = "pm1"    # с чего начинает страница: на −1..+1 функции интереснее

def sombrero(r):
    return 1.0 if r < 1e-9 else math.sin(5 * r) / (5 * r)


# Функции, которым можно учить сеть.
# ключ: (название, {диапазон входов: (формула, функция)}).
# Если диапазона нет в словаре — функция на нём не определена (например, степень на −1..+1).
FUNCTIONS = {
    "mul": ("Умножение", {d: ("x1 × x2", lambda a, b: a * b) for d in DOMAINS}),
    "add": ("Сложение", {d: ("x1 + x2", lambda a, b: a + b) for d in DOMAINS}),
    # обычное x1/x2 улетает в бесконечность при x2 -> 0, поэтому сдвигаем знаменатель,
    # чтобы он не обращался в ноль: на 0..1 делим на (1 + x2), на −1..+1 — на (2 + x2)
    "div": ("Деление", {"01": ("x1 / (1 + x2)", lambda a, b: a / (1.0 + b)),
                        "pm1": ("x1 / (2 + x2)", lambda a, b: a / (2.0 + b))}),
    # XOR для дробных входов: 1, если ровно одно из чисел больше середины диапазона
    "xor": ("XOR-ступенька", {
        "01": ("(x1>0.5) XOR (x2>0.5)", lambda a, b: 1.0 if (a > 0.5) != (b > 0.5) else 0.0),
        "pm1": ("(x1>0) XOR (x2>0)", lambda a, b: 1.0 if (a > 0) != (b > 0) else 0.0)}),
    "absdiff": ("Гладкий XOR", {d: ("|x1 − x2|", lambda a, b: abs(a - b)) for d in DOMAINS}),
    # 0 ** 0 = 1 (как принято в Python и математике), а у оси x1 = 0 функция резко прыгает.
    # На −1..+1 не определена: (−0.5)^0.5 — комплексное число, 0^(−0.5) — бесконечность.
    "pow": ("Степень", {"01": ("x1 ^ x2", lambda a, b: a ** b)}),
    # зависит только от x2 — сеть должна научиться не обращать внимания на x1
    "exp": ("Экспонента", {d: ("e ^ x2", lambda a, b: math.exp(b)) for d in DOMAINS}),
    # --- экзотические: интереснее смотрятся на −1..+1, где центр квадрата в нуле ---
    # круговые волны от центра, как от камня в воде; r = √(x1² + x2²), в центре значение 1
    "hat": ("Сомбреро", {d: ("sin(5r) / (5r)", lambda a, b: sombrero(math.hypot(a, b))) for d in DOMAINS}),
    # шахматный узор из холмов и впадин
    "egg": ("Яичная коробка", {d: ("sin(πx1) · cos(πx2)",
                                    lambda a, b: math.sin(math.pi * a) * math.cos(math.pi * b)) for d in DOMAINS}),
    # светящаяся окружность радиуса 0.6 — замкнутую кривую маленькой сети собрать трудно
    "ring": ("Кольцо", {d: ("e^(−(10·(r − 0.6))²)",
                            lambda a, b: math.exp(-(10 * (math.hypot(a, b) - 0.6)) ** 2)) for d in DOMAINS}),
}
DEFAULT_FUNC = "mul"


def has_domain(func, domain):
    return domain in FUNCTIONS[func][1]


def formula(func, domain=DEFAULT_DOMAIN):
    return FUNCTIONS[func][1][domain][0]


def target_function(x1, x2, func=DEFAULT_FUNC, domain=DEFAULT_DOMAIN):
    """То, чему учим сеть (по умолчанию — произведение двух чисел)."""
    return FUNCTIONS[func][1][domain][1](x1, x2)


def value_range(func, domain=DEFAULT_DOMAIN, n=100):
    """Наименьшее и наибольшее значение функции на квадрате входов (по сетке n×n)."""
    _, lo, hi = DOMAINS[domain]
    pts = [lo + (hi - lo) * k / n for k in range(n + 1)]
    vals = [target_function(a, b, func, domain) for a in pts for b in pts]
    return min(vals), max(vals)


def weights_file(func, domain=DEFAULT_DOMAIN, r2=False):
    """Имя файла весов: для умножения на 0..1 без подсказки — прежний weights.json."""
    if func == DEFAULT_FUNC and domain == DEFAULT_DOMAIN and not r2:
        return "weights.json"
    return (f"weights_{func}" + ("" if domain == DEFAULT_DOMAIN else f"_{domain}")
            + ("_r2" if r2 else "") + ".json")


def make_dataset(n, seed=None, func=DEFAULT_FUNC, domain=DEFAULT_DOMAIN):
    """n случайных примеров ((x1, x2), f(x1, x2)), где x1, x2 из диапазона domain."""
    _, lo, hi = DOMAINS[domain]
    rnd = random.Random(seed)
    data = []
    for _ in range(n):
        x1, x2 = rnd.uniform(lo, hi), rnd.uniform(lo, hi)
        data.append(((x1, x2), target_function(x1, x2, func, domain)))
    return data


class NeuralNetwork:
    def __init__(self, n_inputs=2, n_hidden=8, seed=None, r2_input=False):
        # r2_input — подсказка: к x1, x2 добавляется третий вход r² = x1² + x2²
        # (квадрат расстояния до центра; помогает «круглым» функциям — кольцу, сомбреро)
        self.r2_input = r2_input
        if r2_input:
            n_inputs = 3
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        rnd = random.Random(seed)
        # Инициализация Ксавье (Xavier): случайные веса в разумном диапазоне
        lim1 = math.sqrt(6.0 / (n_inputs + n_hidden))
        lim2 = math.sqrt(6.0 / (n_hidden + 1))
        # w1[j][i] — вес связи от входа i к скрытому нейрону j
        self.w1 = [[rnd.uniform(-lim1, lim1) for _ in range(n_inputs)]
                   for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        # w2[j] — вес связи от скрытого нейрона j к выходу
        self.w2 = [rnd.uniform(-lim2, lim2) for _ in range(n_hidden)]
        self.b2 = 0.0

    # ---------- прямой проход ----------
    def features(self, x):
        """Что на самом деле подаётся на входные нейроны: (x1, x2) или (x1, x2, r²)."""
        if self.r2_input:
            return (x[0], x[1], x[0] * x[0] + x[1] * x[1])
        return tuple(x)

    def forward(self, x):
        """x = (x1, x2). Возвращает (выход, список активаций скрытого слоя)."""
        return self._forward(self.features(x))

    def _forward(self, x):
        h = [sigmoid(sum(w * xi for w, xi in zip(row, x)) + b)
             for row, b in zip(self.w1, self.b1)]
        y = sum(w * hj for w, hj in zip(self.w2, h)) + self.b2
        return y, h

    def predict(self, x1, x2):
        return self.forward((x1, x2))[0]

    # ---------- обучение ----------
    def train_step(self, x, target, lr):
        """Один шаг градиентного спуска на одном примере. Возвращает квадрат ошибки."""
        x = self.features(x)
        y, h = self._forward(x)
        err = y - target  # производная функции потерь 0.5*err^2 по y
        for j in range(self.n_hidden):
            # Ошибка, дошедшая до скрытого нейрона j (считаем ДО изменения w2[j])
            dh = err * self.w2[j] * h[j] * (1.0 - h[j])
            self.w2[j] -= lr * err * h[j]
            for i in range(self.n_inputs):
                self.w1[j][i] -= lr * dh * x[i]
            self.b1[j] -= lr * dh
        self.b2 -= lr * err
        return err * err

    def train_epoch(self, data, lr, rnd=random):
        """Один проход по всем примерам в случайном порядке. Возвращает MSE."""
        order = list(data)
        rnd.shuffle(order)
        total = 0.0
        for x, t in order:
            total += self.train_step(x, t, lr)
        return total / len(order)

    def evaluate(self, data):
        """Среднеквадратичная ошибка (MSE) без обучения."""
        return sum((self.predict(*x) - t) ** 2 for x, t in data) / len(data)

    # ---------- сохранение ----------
    def save(self, path, func=DEFAULT_FUNC, domain=DEFAULT_DOMAIN):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"func": func, "domain": domain, "r2": self.r2_input, "w1": self.w1, "b1": self.b1,
                       "w2": self.w2, "b2": self.b2}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        net = cls(n_inputs=len(d["w1"][0]), n_hidden=len(d["w1"]), r2_input=d.get("r2", False))
        net.w1, net.b1, net.w2, net.b2 = d["w1"], d["b1"], d["w2"], d["b2"]
        net.func = d.get("func", DEFAULT_FUNC)  # старые файлы — умножение
        net.domain = d.get("domain", DEFAULT_DOMAIN)  # и входы 0..1
        return net
