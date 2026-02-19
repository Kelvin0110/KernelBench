import time
import argparse
import statistics
import os
import warnings
from dotenv import load_dotenv
from litellm import completion

# Load environment variables (API keys)
load_dotenv()

# Suppress Pydantic serialization warnings often triggered by DeepSeek/NIM field mismatches
# warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

def test_speed(model_name, num_trials=5):
    # Ensure the model string has the provider prefix
    if not model_name.startswith("nvidia_nim/"):
        full_model_name = f"nvidia_nim/{model_name}"
    else:
        full_model_name = model_name

    print(f"Testing Speed for Model: {full_model_name}")
    print(f"Running {num_trials} trials...\n")

    complex_question = (
        "Two friends, Alice and Bob, are 100 miles apart. Alice starts walking towards Bob at 3 mph, "
        "and Bob starts walking towards Alice at 2 mph. At the same time, a fly starts at Alice's nose "
        "and flies towards Bob at 10 mph. When the fly reaches Bob, it turns around and flies back to Alice. "
        "The fly continues to fly back and forth until Alice and Bob meet. What is the total distance "
        "the fly has traveled? Explain your logic step by step."
    )

    messages = [{"role": "user", "content": complex_question}]
    
    total_times = []
    token_counts = []
    tokens_per_sec = []

    for i in range(num_trials):
        print(f"Trial {i+1}/{num_trials}...", end="", flush=True)
        
        start_time = time.time()
        try:
            response = completion(
                model=full_model_name,
                messages=messages,
                temperature=0.0,  # Keep it deterministic for speed tests
                max_tokens=1024
            )
            # print(response)
            end_time = time.time()
            
            elapsed = end_time - start_time
            total_times.append(elapsed)
            
            # Extract token info
            usage = response.get('usage', {})
            completion_tokens = usage.get('completion_tokens', 0)
            token_counts.append(completion_tokens)
            
            if completion_tokens > 0:
                tps = completion_tokens / elapsed
                tokens_per_sec.append(tps)
                print(f" Done ({elapsed:.2f}s, {completion_tokens} tokens, {tps:.2f} tokens/s)")
            else:
                tokens_per_sec.append(0)
                print(f" Done ({elapsed:.2f}s, no token usage reported)")
                
        except Exception as e:
            print(f" Failed! Error: {e}")

    if not total_times:
        print("\nAll trials failed. Please check your API key and model name.")
        return

    # Calculate statistics
    avg_time = statistics.mean(total_times)
    std_time = statistics.stdev(total_times) if len(total_times) > 1 else 0
    avg_tokens = statistics.mean(token_counts)
    avg_tps = statistics.mean(tokens_per_sec) if tokens_per_sec else 0

    print("\n" + "="*50)
    print("NVIDIA NIM SPEED TEST SUMMARY")
    print("="*50)
    print(f"Model: {full_model_name}")
    print(f"Trials: {len(total_times)}")
    print(f"Avg Response Time: {avg_time:.3f} s (± {std_time:.3f} s)")
    print(f"Avg Completion Tokens: {avg_tokens:.1f}")
    print(f"Avg Tokens / Sec: {avg_tps:.2f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NVIDIA NIM model speed using LiteLLM")
    parser.add_argument("model", type=str, help="Model name (e.g., deepseek-ai/deepseek-r1-0528 or meta/llama-3.1-70b-instruct)")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials (default: 5)")
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.environ.get("NVIDIA_NIM_API_KEY") and not os.environ.get("NVIDIA_API_KEY"):
        print("Warning: Neither NVIDIA_NIM_API_KEY nor NVIDIA_API_KEY found in environment.")
        print("Make sure it is set in your .env file or export it.")
        print("-" * 30)

    test_speed(args.model, args.trials)
