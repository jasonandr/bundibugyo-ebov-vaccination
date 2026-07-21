import argparse
import csv

from run_final_high_replicate_estimates import BASE_SEED, RAW_FIELDS, N_REPLICATES, run_one, scenario_definitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-index", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    scenario = scenario_definitions()[args.scenario_index]
    rows = []
    for replicate in range(args.start, args.end):
        print(f"Running replicate {replicate}", flush=True)
        seed = 42 + args.scenario_index * 1000000 + replicate
        rows.append(run_one((scenario, replicate, seed)))

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
