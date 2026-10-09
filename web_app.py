"""Веб-визуализация обучения и работы нейронной сети (x1*x2 и другие функции).

Запуск:  python3 web_app.py
Откроется браузер с адресом http://127.0.0.1:8765
Сеть обучается здесь, в Python (network.py), а страница web/index.html
только рисует её состояние. Остановить сервер: Ctrl+C.
"""
import json
import os
import random
import threading
import time
import webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from network import (DEFAULT_DOMAIN, DEFAULT_FUNC, DOMAINS, FUNCTIONS, NeuralNetwork, formula, has_domain,
                     make_dataset, target_function, value_range, weights_file)

HOST, PORT = "127.0.0.1", 8765
HIDDEN = 8          # нейронов в скрытом слое
SAMPLES = 200       # обучающих примеров
GRID = 30           # разрешение цветных карт и объёмного графика
BASE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE, "web")


class Trainer:
    """Хранит сеть и обучает её в фоновом потоке."""

    def __init__(self):
        self.lock = threading.Lock()
        self.func = DEFAULT_FUNC
        self.domain = DEFAULT_DOMAIN
        self.r2 = False         # подсказка: третий вход r² = x1² + x2²
        self.lr = 0.3
        self.speed = 2          # эпох за шаг
        self.running = False
        self.message = ""
        self.reset()
        threading.Thread(target=self.loop, daemon=True).start()

    def reset(self):
        with self.lock:
            # Данные пересчитываются под текущую функцию
            self.train_data = make_dataset(SAMPLES, seed=1, func=self.func, domain=self.domain)
            self.test_data = make_dataset(100, seed=1001, func=self.func, domain=self.domain)
            self.range = value_range(self.func, self.domain)
            # правильные ответы на сетке карт — не меняются, пока не сменили функцию или входы
            self.true_grid = [[target_function(a, b, self.func, self.domain) for a in self.grid_points()]
                              for b in self.grid_points()]
            self.net = NeuralNetwork(n_hidden=HIDDEN, seed=random.randrange(10 ** 6), r2_input=self.r2)
            self.rnd = random.Random()
            self.epoch = 0
            self.hist_train, self.hist_test = [], []
            self.loaded = False

    def grid_points(self):
        """Центры клеток карты по одной оси."""
        _, lo, hi = DOMAINS[self.domain]
        return [lo + (hi - lo) * (k + 0.5) / GRID for k in range(GRID)]

    def loop(self):
        while True:
            if self.running:
                with self.lock:
                    for _ in range(self.speed):
                        mse = self.net.train_epoch(self.train_data, self.lr, self.rnd)
                        self.epoch += 1
                        self.hist_train.append(mse)
                        self.hist_test.append(self.net.evaluate(self.test_data))
                        if mse != mse or mse > 1e6:  # NaN или взрыв — скорость обучения слишком велика
                            self.running = False
                            self.message = "Ошибка разошлась — уменьшите скорость обучения и нажмите «Сброс»"
                            break
            time.sleep(0.03)

    def state(self, since):
        with self.lock:
            net, func, domain = self.net, self.func, self.domain
            _, lo, hi = DOMAINS[domain]
            pts = self.grid_points()
            grid = [[net.predict(a, b) for a in pts] for b in pts]  # grid[j][i]: j — x2, i — x1
            return {
                "func": func, "domain": domain, "inputs": [lo, hi], "range": self.range,
                "formula": formula(func, domain), "file": weights_file(func, domain, net.r2_input),
                "r2": net.r2_input,
                "domains": [{"key": k, "name": d[0]} for k, d in DOMAINS.items()],
                "funcs": [{"key": k, "name": f[0], "available": has_domain(k, domain),
                           "formula": formula(k, domain) if has_domain(k, domain) else ""}
                          for k, f in FUNCTIONS.items()],
                "epoch": self.epoch, "running": self.running, "lr": self.lr, "speed": self.speed,
                "loaded": self.loaded, "message": self.message,
                "since": since, "train": self.hist_train[since:], "test": self.hist_test[since:],
                "w1": net.w1, "b1": net.b1, "w2": net.w2, "b2": net.b2,
                "grid": grid, "true_grid": self.true_grid,
            }

    def predict(self, x1, x2):
        """Ответ сети вместе с весами, по которым он посчитан (для таблицы расчёта)."""
        with self.lock:
            net = self.net
            y, h = net.forward((x1, x2))
            w1, b1 = [row[:] for row in net.w1], net.b1[:]
            w2, b2 = net.w2[:], net.b2
        return {"y": y, "hidden": h, "target": target_function(x1, x2, self.func, self.domain),
                "w1": w1, "b1": b1, "w2": w2, "b2": b2}

    def control(self, cmd):
        action = cmd.get("action")
        self.message = ""
        if "lr" in cmd:
            self.lr = min(1.0, max(0.001, float(cmd["lr"])))
        if "speed" in cmd:
            self.speed = min(50, max(1, int(cmd["speed"])))
        if action == "func" and cmd.get("func") in FUNCTIONS and has_domain(cmd["func"], self.domain):
            self.running = False
            self.func = cmd["func"]
            self.reset()
        elif action == "r2":
            self.running = False
            self.r2 = bool(cmd.get("on"))
            self.reset()
        elif action == "domain" and cmd.get("domain") in DOMAINS:
            self.running = False
            self.domain = cmd["domain"]
            if not has_domain(self.func, self.domain):  # степень на −1..+1 не определена
                self.message = f"«{FUNCTIONS[self.func][0]}» не определена для входов {DOMAINS[self.domain][0]} — выбрано умножение"
                self.func = DEFAULT_FUNC
            self.reset()
        elif action == "start":
            self.running = True
        elif action == "pause":
            self.running = False
        elif action == "reset":
            self.running = False
            self.reset()
        elif action == "save":
            with self.lock:
                name = weights_file(self.func, self.domain, self.net.r2_input)
                self.net.save(os.path.join(BASE, name), self.func, self.domain)
            self.message = f"Веса сохранены в {name}"
        elif action == "load":
            name = weights_file(self.func, self.domain, self.r2)
            path = os.path.join(BASE, name)
            if not os.path.exists(path):
                self.message = (f"Файл {name} не найден — сначала сохраните веса "
                                f"или запустите python3 train.py --func {self.func} --domain {self.domain}"
                                + (" --r2" if self.r2 else ""))
            else:
                self.running = False
                self.reset()
                with self.lock:
                    self.net = NeuralNetwork.load(path)
                    self.loaded = True
                self.message = f"Веса загружены из {name}"


trainer = Trainer()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def send_json(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/api/state":
            self.send_json(trainer.state(int(q.get("since", ["0"])[0])))
        elif url.path == "/api/predict":
            self.send_json(trainer.predict(float(q["x1"][0]), float(q["x2"][0])))
        else:
            super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/control":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        trainer.control(json.loads(self.rfile.read(length) or b"{}"))
        self.send_json({"ok": True})

    def log_message(self, *args):
        pass  # не засорять консоль запросами


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"Сервер запущен: {url}\nОстановить: Ctrl+C")
    threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен")


if __name__ == "__main__":
    main()
