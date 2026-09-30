#!/usr/bin/env python3
"""
Find the best epoch for Wind_3e-4 and Humidity_3e-4 from training log
"""
import re
from statistics import median

def extract_epoch_kges(log_file, variation_name):
    """Extract validation KGE scores for each epoch"""

    with open(log_file, 'r') as f:
        content = f.read()

    # Find the section for this variation
    pattern = f"VARIATION.*{variation_name}.*?(?=VARIATION|ALL TRAINING COMPLETE|$)"
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print(f"Could not find {variation_name} in log")
        return None

    variation_content = match.group(0)

    # Extract KGE scores by epoch
    epoch_kges = {}

    # Find all epochs
    for epoch_num in range(30):
        # Find content for this epoch
        epoch_pattern = f"Epoch {epoch_num}:.*?Checkpoint saved.*epoch{epoch_num}.pth"
        epoch_match = re.search(epoch_pattern, variation_content, re.DOTALL)

        if epoch_match:
            epoch_content = epoch_match.group(0)

            # Extract all test_kge values (these are validation scores during training)
            kge_pattern = r'test_kge:\s+([-\d.]+)'
            kges = [float(k) for k in re.findall(kge_pattern, epoch_content)]

            if kges:
                epoch_kges[epoch_num] = kges
                print(f"  Epoch {epoch_num}: {len(kges)} validation scores")

    return epoch_kges

def find_best_epoch(epoch_kges, variation_name):
    """Calculate median KGE per epoch and find the best"""

    print(f"\n{'='*70}")
    print(f"{variation_name} - Per-Epoch Median Validation KGE")
    print(f"{'='*70}")

    if not epoch_kges:
        print("No epoch data found!")
        return None

    results = []
    for epoch_num in sorted(epoch_kges.keys()):
        kges = epoch_kges[epoch_num]
        med = median(kges)
        mean_val = sum(kges) / len(kges)
        results.append({
            'epoch': epoch_num,
            'median': med,
            'mean': mean_val,
            'min': min(kges),
            'max': max(kges),
            'count': len(kges)
        })

    # Print all epochs
    print(f"\n{'Epoch':<8} {'Median KGE':<12} {'Mean KGE':<12} {'Min':<10} {'Max':<10} {'Count':<8}")
    print("-" * 70)
    for r in results:
        print(f"{r['epoch']:<8} {r['median']:<12.4f} {r['mean']:<12.4f} {r['min']:<10.4f} {r['max']:<10.4f} {r['count']:<8}")

    # Find best epoch
    best = max(results, key=lambda x: x['median'])

    print(f"\n{'='*70}")
    print(f"🏆 BEST EPOCH: {best['epoch']}")
    print(f"   Median KGE: {best['median']:.4f}")
    print(f"   Mean KGE: {best['mean']:.4f}")
    print(f"   Range: {best['min']:.4f} to {best['max']:.4f}")
    print(f"{'='*70}")

    return best

# Main execution
if __name__ == "__main__":
    log_file = "/Users/simran/Desktop/SnowML/training_final.log"

    print("Analyzing training log...\n")

    # Analyze Wind_3e-4
    print("Extracting Wind_3e-4 validation scores...")
    wind_kges = extract_epoch_kges(log_file, "Wind_3e-4")
    wind_best = find_best_epoch(wind_kges, "Wind_3e-4")

    print("\n" + "="*70 + "\n")

    # Analyze Humidity_3e-4
    print("Extracting Humidity_3e-4 validation scores...")
    humidity_kges = extract_epoch_kges(log_file, "Humidity_3e-4")
    humidity_best = find_best_epoch(humidity_kges, "Humidity_3e-4")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - Best Checkpoints to Download")
    print("="*70)
    if wind_best:
        print(f"Wind_3e-4:")
        print(f"  File: Exp1B_Wind_3e-4_epoch{wind_best['epoch']}.pth")
        print(f"  Median KGE: {wind_best['median']:.4f}")

    if humidity_best:
        print(f"\nHumidity_3e-4:")
        print(f"  File: Exp1B_Humidity_3e-4_epoch{humidity_best['epoch']}.pth")
        print(f"  Median KGE: {humidity_best['median']:.4f}")

    print("\nExpected from original Exp3: Median KGE ~0.82")
    print("="*70)
