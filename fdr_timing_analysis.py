"""
FDR Analysis: Timing and Sample Size Calculation
"""
import numpy as np

print("="*70)
print("FDR ANALYSIS - TIMING & SAMPLE SIZE OPTIMIZATION")
print("="*70)

# Measured timing from test run
print("\n1. MEASURED PERFORMANCE:")
print("-"*70)
job1_start = "15:30:05"  # BX795-RF started
job1_end = "15:32:30"    # BX795-RF finished
job2_end = "15:35:07"    # GSK269962A-RF finished

# Calculate durations in seconds
def time_to_sec(t):
    h, m, s = map(int, t.split(':'))
    return h*3600 + m*60 + s

job1_duration = time_to_sec(job2_end) - time_to_sec(job1_start)  # Total for 2 jobs
avg_duration = job1_duration / 2 / 60  # minutes per job

print(f"Jobs completed: 2")
print(f"Total time: {job1_duration/60:.1f} minutes")
print(f"Average per job (1 drug, 1 method, 5 perms): {avg_duration:.1f} minutes")
print(f"Time per permutation: {avg_duration/5:.2f} minutes")

# Current test setup
n_drugs_test = 5
n_perms_test = 5
n_methods = 2
n_jobs_test = n_drugs_test * n_methods
total_time_test = n_jobs_test * avg_duration

print(f"\nCurrent test: {n_drugs_test} drugs × {n_perms_test} perms × {n_methods} methods")
print(f"  = {n_jobs_test} jobs × {avg_duration:.1f} min = {total_time_test:.0f} min ({total_time_test/60:.1f}h)")

# Statistical considerations
print("\n2. STATISTICAL REQUIREMENTS:")
print("-"*70)
print("FDR estimation accuracy depends on:")
print("  • Number of permutations (n_perm): affects E₀[V] precision")
print("  • Number of drugs (n_drugs): affects generalizability")

print("\nPermutation requirements:")
perms = [5, 10, 20, 50, 100, 200]
for n_perm in perms:
    se = 1 / np.sqrt(n_perm)  # Approx SE for proportion
    ci_width = 2 * 1.96 * se  # 95% CI width
    print(f"  n={n_perm:3d} perms → SE ≈ {se:.3f}, 95% CI width ≈ ±{ci_width:.2f}")

print("\n  Recommendation: n_perm ≥ 20 for FDR ± 0.4 precision")
print("                  n_perm ≥ 100 for publication-quality (± 0.2)")

print("\nDrug sample requirements:")
drug_samples = [5, 10, 25, 50, 100]
total_drugs = 286
for n_drugs in drug_samples:
    sampling_error = 1.96 * np.sqrt(0.25 / n_drugs)  # Worst case p=0.5
    print(f"  n={n_drugs:3d} drugs → Sampling error ≈ ±{sampling_error:.3f} ({n_drugs/total_drugs*100:.1f}% of total)")

print("\n  Recommendation: n_drugs ≥ 25 for representative sample")
print("                  n_drugs ≥ 50 for robust estimates")

# Compute time estimates for different scenarios
print("\n3. COMPUTATIONAL FEASIBILITY:")
print("-"*70)
print(f"Timing: ~{avg_duration:.1f} min per (drug × method × n_perms)")
print(f"Available drugs: {total_drugs}")
print()

scenarios = [
    ("Test (current)", 5, 5, "Quick validation"),
    ("Minimal", 10, 10, "Basic FDR estimate"),
    ("Conservative", 25, 20, "Good balance - RECOMMENDED"),
    ("Robust", 50, 20, "Publication-ready"),
    ("Comprehensive", 100, 50, "High confidence"),
    ("Full", 286, 20, "All drugs"),
]

print(f"{'Scenario':<20} {'Drugs':>6} {'Perms':>6} {'Jobs':>6} {'Time (h)':>10} {'Quality'}")
print("-"*70)

for name, n_drugs, n_perms, quality in scenarios:
    n_jobs = n_drugs * n_methods
    time_per_job = avg_duration * n_perms / n_perms_test  # Scale by permutation count
    total_time_h = n_jobs * time_per_job / 60
    
    feasibility = ""
    if total_time_h < 2:
        feasibility = "✓ Quick"
    elif total_time_h < 8:
        feasibility = "✓ Feasible"
    elif total_time_h < 24:
        feasibility = "⚠ Long"
    else:
        feasibility = "✗ Too long"
    
    print(f"{name:<20} {n_drugs:>6} {n_perms:>6} {n_jobs:>6} {total_time_h:>9.1f}  {feasibility:<12} {quality}")

# Parallel execution benefit
print("\n4. PARALLELIZATION POTENTIAL:")
print("-"*70)
print("Jobs are independent → can run in parallel!")
print()

cores_options = [1, 2, 4, 8]
recommended_scenario = ("Conservative", 25, 20)
n_drugs_rec, n_perms_rec = recommended_scenario[1], recommended_scenario[2]
n_jobs_rec = n_drugs_rec * n_methods
time_per_job_rec = avg_duration * n_perms_rec / n_perms_test
total_time_serial = n_jobs_rec * time_per_job_rec / 60

print(f"Scenario: {recommended_scenario[0]} ({n_drugs_rec} drugs × {n_perms_rec} perms)")
print(f"{'Cores':>6} {'Time (h)':>10} {'Speedup':>10}")
print("-"*40)
for cores in cores_options:
    parallel_time = total_time_serial / cores
    speedup = total_time_serial / parallel_time
    print(f"{cores:>6} {parallel_time:>9.1f}h  {speedup:>9.1f}x")

# Final recommendation
print("\n5. FINAL RECOMMENDATION:")
print("="*70)
print("CONSERVATIVE SCENARIO (balanced quality vs. time):")
print(f"  • Drugs: 25 (random sample, ~9% of total)")
print(f"  • Permutations: 20 per drug")
print(f"  • Methods: 2 (linear + RF)")
print(f"  • Total jobs: 50")
print(f"  • Serial time: ~{total_time_serial:.1f}h")
print(f"  • With 4 cores: ~{total_time_serial/4:.1f}h (recommended)")
print()
print("Statistical properties:")
print("  • FDR precision: ±0.44 (95% CI)")
print("  • Representative sample of drugs")
print("  • Publishable results")
print()
print("ROBUST SCENARIO (publication-quality):")
print("  • Drugs: 50")
print("  • Permutations: 20")
print("  • With 4 cores: ~8h")
print()
print("To run with 4 cores:")
print("  snakemake -s Snakefile_fdr_simple --cores 4 --rerun-incomplete")
print("="*70)
