import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from conversion import test_agent
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-conversions", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--agent-type", type=str, default="code")
    return parser.parse_args()


def wrap_test_agent(is_code_agent: bool, n_conversions: int, step_by_step: bool, i: int):
    return {f"{n_conversions}_{i}": test_agent(is_code_agent, n_conversions, step_by_step, silent=True)}


if __name__ == "__main__":
    args = parse_args()
    print(f"Starting run with arguments: {args}")

    is_code_agent = args.agent_type == "code"
    step_by_step = True

    with ThreadPoolExecutor(max_workers=64) as exe:
        results: dict[int, list[bool]] = {}
        futures: list = []
        for n_conversions in range(10, args.n_conversions + 1, 10):
            results[n_conversions] = []
            for i in range(args.n_samples):
                futures.append(exe.submit(wrap_test_agent, is_code_agent, n_conversions, step_by_step, i))

        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
            outcomes = f.result()
            for name, result in outcomes.items():
                n_conversions = int(name.split("_")[0])
                results[n_conversions].append(result if result is not None else False)

    with open(f"benchmark_results_{args.agent_type}.json", "w") as f:
        json.dump(results, f)

    print("All tasks processed.")
