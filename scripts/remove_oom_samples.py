import json
import argparse
import os

def remove_oom_items(run_name, runs_dir="/home/kwtamai/KernelBench/runs"):
    eval_results_path = os.path.join(runs_dir, run_name, "eval_results.json")
    
    if not os.path.exists(eval_results_path):
        print(f"Error: {eval_results_path} not found.")
        return

    with open(eval_results_path, 'r') as f:
        data = json.load(f)

    cleaned_data = {}
    removed_count = 0

    for problem_id, samples in data.items():
        if not isinstance(samples, list):
            cleaned_data[problem_id] = samples
            continue
            
        new_samples = []
        for sample in samples:
            # Check for OOM in metadata or directly in sample if it exists there
            metadata = sample.get('metadata', {})
            runtime_error = metadata.get('runtime_error', '')
            
            # The user example shows "CUDA out of memory" in runtime_error
            # We also check runtime_error_traceback just in case
            traceback = metadata.get('runtime_error_traceback', '')
            
            is_oom = "CUDA out of memory" in runtime_error or "CUDA out of memory" in traceback
            
            if is_oom:
                removed_count += 1
                continue
            else:
                new_samples.append(sample)
        
        # If all samples for this problem were removed (e.g., all OOM),
        # skip adding the problem key so the key and its value are removed.
        if new_samples:
            cleaned_data[problem_id] = new_samples

    if removed_count > 0:
        with open(eval_results_path, 'w') as f:
            json.dump(cleaned_data, f, indent=4)
        print(f"Successfully removed {removed_count} OOM items from {eval_results_path}")
    else:
        print("No OOM items found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove samples with CUDA OOM from eval_results.json")
    parser.add_argument("run_name", help="Name of the run folder in the runs/ directory")
    args = parser.parse_args()
    
    remove_oom_items(args.run_name)
