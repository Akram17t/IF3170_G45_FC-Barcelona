import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .algorithms.common import DEFAULTS, HC_DEFAULTS
from .algorithms.genetic import genetic_algorithm
from .algorithms.hillclimbing import VARIANTS
from .algorithms.sa import simulated_annealing
from .model import Problem, load_problem
from .results import check_result, sa_directory, save_result
from .visualization import render_run

ROOT = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).with_name("web") / "index.html"
FONTS = {"FCBARCELONAPLAYSOLID.woff2": "font/woff2", "FCBARCELONAPLAYSOLID.woff": "font/woff",
         "FCBARCELONAPLAYSOLID.ttf": "font/ttf"}


def defaults():
    # HC beda parameter tiap varian, jadi digabung dulu di sini
    hc = {}
    for variant, extra in HC_DEFAULTS.items():
        hc[variant] = {**DEFAULTS["hc"], **extra}
    return {"hc": hc, "sa": DEFAULTS["sa"], "ga": DEFAULTS["ga"]}


def cases():
    found = []
    for path in sorted((ROOT / "examples").glob("*.json")):
        found.append({"name": path.stem, "problem": load_problem(path).to_dict()})
    return found


def save_directory(body, result):
    # SA dipisah per kasus, seed, dan run biar hasil eksperimen ga ketimpa
    if body["algorithm"] == "ga":
        config = result.config
        return ROOT / "output" / "ga" / f"pop{config['population_size']}-gen{config['generations']}" / f"seed{result.seed}"
    if body["algorithm"] == "hc":
        return ROOT / "output" / "hc" / body["variant"] / f"seed{result.seed}"
    problem = Problem.from_dict(body["problem"])
    return sa_directory(ROOT / "output" / "sa", problem, body.get("case_name", "custom"), result.seed)


def run(body):
    problem = Problem.from_dict(body["problem"])
    algorithm = body["algorithm"]
    config = body.get("config", {})

    if algorithm == "hc":
        if body.get("variant") not in VARIANTS:
            raise ValueError("Varian HC tidak dikenal")
        result = VARIANTS[body["variant"]](problem, config)
    elif algorithm == "sa":
        result = simulated_annealing(problem, config)
    elif algorithm == "ga":
        result = genetic_algorithm(problem, config)
    else:
        raise ValueError("Algoritma harus hc, sa, atau ga")

    if algorithm == "sa":
        check_result(problem, result)
    saved = None
    if body.get("save"):
        directory = save_directory(body, result)
        directory.mkdir(parents=True, exist_ok=True)
        save_result(directory / "result.json", problem, result, body.get("case_name", "custom"))
        render_run(problem, result, directory, body.get("case_name", "custom"))
        saved = str(directory.relative_to(ROOT))

    # Metrik yang berupa angka/teks saja, daftar panjangnya diambil yang dipakai grafik
    metrics = {}
    for key, value in result.metrics.items():
        if isinstance(value, (int, float, str, bool)):
            metrics[key] = value

    data = dict(
        algorithm=result.algorithm, seed=result.seed, config=dict(result.config), metrics=metrics,
        initial_score=result.initial_score, final_score=result.final_score,
        best_score=result.best_score, duration_seconds=result.duration_seconds,
        stop_reason=result.stop_reason, objective_history=list(result.objective_history),
        initial_state=result.initial_state.to_dict(), final_state=result.final_state.to_dict(),
        best_state=result.best_state.to_dict(), saved=saved,
    )
    if algorithm == "ga":
        data["mean_history"] = result.metrics["mean_history"]
    if algorithm == "sa":
        data["temperature"] = [record["temperature"] for record in result.metrics["records"]]
        data["records"] = result.metrics["records"]
    return data


class Handler(BaseHTTPRequestHandler):
    def send(self, status, body, kind="application/json"):
        if kind == "application/json":
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.send(200, PAGE.read_bytes(), "text/html")
        elif self.path.startswith("/font/") and self.path[6:] in FONTS:
            file = ROOT / "FCBARCELONAPLAYSOLID" / self.path[6:]
            if file.is_file():
                self.send(200, file.read_bytes(), FONTS[self.path[6:]])
            else:
                self.send(404, {"error": "font tidak ada"})
        elif self.path == "/api/cases":
            self.send(200, cases())
        elif self.path == "/api/defaults":
            self.send(200, defaults())
        else:
            self.send(404, {"error": "tidak ditemukan"})

    def do_POST(self):
        if self.path != "/api/run":
            self.send(404, {"error": "tidak ditemukan"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send(200, run(body))
        except (ValueError, KeyError) as error:
            self.send(400, {"error": f"{type(error).__name__}: {error}"})


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Halaman web interaktif untuk HC, SA, dan GA")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    address = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Buka {address} (Ctrl+C untuk berhenti)")
    webbrowser.open(address)
    server.serve_forever()


if __name__ == "__main__":
    main()
