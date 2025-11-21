"""
Bayesian Optimization with ngspice (single simulation per sizing)
ThreadPool version — at most 10 concurrent simulations (dirs 1–10)
"""

import os
from random import random
import re
import time
import subprocess
import numpy as np
import torch
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
SEED = 2025  
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def doSimulation(myparams,spec_targets):


# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# ==========================
# Parameter bounds
# ==========================


    PARAM_ORDER = list(myparams.keys())

    # ==========================
    # Spec targets
    # ==========================


    # ==========================
    # Helper functions
    # ==========================
    def build_bounds_from_myparams(mp: dict):
        lb = torch.tensor([lo for (lo, hi) in mp.values()], dtype=torch.double)
        ub = torch.tensor([hi for (lo, hi) in mp.values()], dtype=torch.double)
        return torch.stack([lb, ub])


    def tensor_to_params_dict_from_myparams(t: torch.Tensor, order=PARAM_ORDER):
        vals = t.tolist()
        d = {}
        for i, k in enumerate(order):
            v = float(vals[i])
            if k.endswith("_M"):
                v = max(1, int(round(v)))
            d[k] = v
        return d


    def write_varspice(workdir, params: dict, order=PARAM_ORDER):
        os.makedirs(workdir, exist_ok=True)
        lines = [".PARAM"]
        kvs = [f"{k}={params[k]}" for k in order]
        for i in range(0, len(kvs), 3):
            lines.append("+ " + " ".join(kvs[i:i + 3]))
        with open(os.path.join(workdir, "var.spice"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


    def run_ngspice(workdir):
        start_time = time.time()
        cmd = ["ngspice", "-b", "test.spice", "-o", "result.txt"]
        try:
            subprocess.run(cmd, cwd=workdir, check=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        print(f"[{workdir}] Ngspice done in {time.time() - start_time:.2f}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def analyseResult(workdir):
        params = [
            "dcgain_", "gain_bandwidth_product_", "phase_margin",
            "ivdd25", "cmrrdc", "dcpsrp", "dcpsrn"
        ]
        results = {}
        try:
            with open(os.path.join(workdir, "result.txt"), "r", encoding="utf-8") as f:
                log_text = f.read()
        except FileNotFoundError:
            print(f"[{workdir}] Missing result.txt")
            return {}

        for param in params:
            match = re.search(rf"{param}\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", log_text)
            results[param] = float(match.group(1)) if match else None

        try:
            results["Power"] = (
                results["ivdd25"] * 1.8 * 1e6 * (-1)
                if results.get("ivdd25") is not None else 1000
            )
        except Exception:
            results["Power"] = 1000

        if "gain_bandwidth_product_" in results:
            results["GBW"] = results["gain_bandwidth_product_"]
        if "dcpsrn" in results:
            results["PSRN"] = abs(results.get("dcpsrn") or 0)
        if "dcpsrp" in results:
            results["PSRP"] = abs(results.get("dcpsrp") or 0)
        if "phase_margin" in results:
            results["phase_margin (deg)"] = results.get("phase_margin")
        if "dcgain_" in results:
            results["dcgain"] = results.get("dcgain_")
        if "cmrrdc" in results:
            results["cmrrdc"] = abs(results.get("cmrrdc") or 0)

        print(f"[{workdir}] Results: {results}")
        return results


    def extract_specs(workdir):
        return analyseResult(workdir)


    def reward_func(specs: dict, targets: dict) -> float:
        reward = 0.0
        for key, cond in targets.items():
            if key not in specs or specs[key] is None:
                reward -= 100.0
                continue
            val = specs[key]
            op, goal = list(cond.items())[0]
            diff_ratio = min(abs((val - goal) / (abs(goal) + 1e-9)), 5.0)
            ok = (op == ">=" and val >= goal) or (op == "<=" and val <= goal) or \
                (op == ">" and val > goal) or (op == "<" and val < goal)
            reward += 1.0 if ok else -diff_ratio
        return reward


    def check_spec_targets(results: dict, targets: dict):
        ans = True
        for k, cond in targets.items():
            if k not in results or results[k] is None:
                print(f"  [!] Missing {k}")
                ans = False
                continue
            op, val = list(cond.items())[0]
            cur = results[k]
            ok = (op == ">=" and cur >= val) or (op == "<=" and cur <= val) or \
                (op == ">" and cur > val) or (op == "<" and cur < val)
            if not ok:
                print(f"  [x] {k}={cur} fails {op} {val}")
                ans = False
        return ans


    def get_next_points(X, Y, best_Y, bounds, q):
        X = X.to(device)
        Y = Y.to(device)
        best_Y = torch.tensor(best_Y, dtype=torch.double, device=device)
        bounds = bounds.to(device)
        X_norm = normalize(X, bounds)
        model = SingleTaskGP(X_norm, Y).to(device)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll,device=device)
        EI = qLogExpectedImprovement(model, best_f=best_Y)
        cand_norm, _ = optimize_acqf(
            EI,
            bounds=torch.stack([
                torch.zeros(bounds.shape[1], dtype=torch.double,device=device),
                torch.ones(bounds.shape[1], dtype=torch.double,device=device)
            ]),
            q=q, num_restarts=10, raw_samples=128
        )
        return unnormalize(cand_norm, bounds)


    # ==========================
    # Concurrency control
    # ==========================
    MAX_WORKERS = 10
    DIR_LOCK = threading.Lock()
    DIR_AVAILABLE = [str(i) for i in range(1, MAX_WORKERS + 1)]


    def acquire_dir():
        while True:
            with DIR_LOCK:
                if DIR_AVAILABLE:
                    d = DIR_AVAILABLE.pop(0)
                    return d
            time.sleep(0.1)


    def release_dir(d):
        with DIR_LOCK:
            if d not in DIR_AVAILABLE:
                DIR_AVAILABLE.append(d)


    def worker(param):
        workdir = acquire_dir()
        try:
            p_dict = tensor_to_params_dict_from_myparams(param, PARAM_ORDER)
            write_varspice(workdir, p_dict)
            mytime = time.time()
            run_ngspice(workdir)
            print(f"[{workdir}] Total sim time: {time.time() - mytime:.2f}s")
            specs = extract_specs(workdir)
            reward = reward_func(specs, spec_targets)
        finally:
            release_dir(workdir)
        return reward


    def target_function(X):
        rewards = [None] * len(X)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(worker, X[i]): i for i in range(len(X))}
            for future in as_completed(futures):
                idx = futures[future]
                rewards[idx] = future.result()
        return torch.tensor(rewards, dtype=torch.double)


    def generate_initial_data(n, bounds):
        lb, ub = bounds
        lb = lb.numpy()
        ub = ub.numpy()
        dim = lb.shape[0]
        X = torch.empty((n, dim), dtype=torch.double)
        for d in range(dim):
            X[:, d] = torch.tensor(np.random.uniform(lb[d], ub[d], size=n), dtype=torch.double)
        Y = target_function(X).unsqueeze(-1)
        best_val = Y.max().item()
        best_idx = torch.argmax(Y).item()
        best_x = X[best_idx, :]
        return X, Y, best_val, best_x


    # ==========================
    # Main optimization loop
    # ==========================
    if __name__ == "__main__":
        bounds = build_bounds_from_myparams(myparams)
        X, Y, best_Y, best_X = generate_initial_data(100, bounds)
        X = X.to(device)
        Y = Y.to(device)
        best_X = best_X.to(device)
        print("Init best reward:", best_Y)

        MAX_ITERS = 100
        for i in range(1, MAX_ITERS + 1):
            start_time = time.time()
            C = get_next_points(X, Y, best_Y, bounds, q=20).to(device)
            Y_new = target_function(C).unsqueeze(-1).to(device)
            X = torch.cat([X, C]).to(device)
            Y = torch.cat([Y, Y_new]).to(device)

            best_idx = torch.argmax(Y).item()
            best_Y = Y[best_idx].item()
            best_X = X[best_idx, :]

            write_varspice("1", tensor_to_params_dict_from_myparams(best_X, PARAM_ORDER))
            run_ngspice("1")
            results = extract_specs("1")
            reward_val = reward_func(results, spec_targets)

            print(f"\nIter {i}: duration {time.time() - start_time:.2f}s")
            print(f" Best Reward (observed): {best_Y:.3f}")
            print(f" True Reward: {reward_val:.3f}")
            for k, v in results.items():
                print(f"  {k}: {v}")
            print(f"  Params: {tensor_to_params_dict_from_myparams(best_X, PARAM_ORDER)}")
            if check_spec_targets(results, spec_targets):
                print("========== All Spec Targets Achieved! ==========")
                break

        print("\n========== Optimization Finished ==========")
        print(f"Best Reward: {best_Y:.3f}")
        print("Best Parameters:")
        print(best_X.numpy())
