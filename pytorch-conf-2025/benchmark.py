"""
Benchmark runner that executes test scripts and aggregates performance metrics.

Usage:
    python benchmark.py --help
    python benchmark.py --case grayscale
    python benchmark.py --case grayscale --script grayscale.py --save results
"""

import argparse
import subprocess
import re
import pandas as pd
import sys
import threading
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def parse_gbps_output(stdout: str) -> pd.DataFrame:
    """
    Parse stdout for lines matching 'shape = (x, x) gbps = xxx' or 'size = xxx gbps = xxx' patterns.
    Supports various formats: shape/size, with/without parentheses, single/dual dimensions.

    Args:
        stdout: The stdout string from running the benchmark script

    Returns:
        DataFrame with columns: shape, gbps
    """
    # Pattern 1: shape = (128, 128) gbps = 85.29807013376632 [GB/s]
    pattern1 = r'shape\s*=\s*\((\d+),\s*(\d+)\)\s+(?:gbps|GBPS)\s*=\s*([\d.]+)(?:\s+(?:GB/s|GiB/s))?'

    # Pattern 2: size = (128, 128) gbps = 85.29807013376632 [GB/s]
    pattern2 = r'size\s*=\s*\((\d+),\s*(\d+)\)\s+(?:gbps|GBPS)\s*=\s*([\d.]+)(?:\s+(?:GB/s|GiB/s))?'

    # Pattern 3: size = 1024 gbps = 85.29807013376632 [GB/s]
    pattern3 = r'size\s*=\s*(\d+)\s+(?:gbps|GBPS)\s*=\s*([\d.]+)(?:\s+(?:GB/s|GiB/s))?'

    # Pattern 4: size = 1024, 1024 gbps = 85.29807013376632
    pattern4 = r'size\s*=\s*(\d+),\s*(\d+)\s+(?:gbps|GBPS)\s*=\s*([\d.]+)(?:\s+(?:GB/s|GiB/s))?'

    # Pattern 5: shape = Shape(128, 128) gbps = 85.29... (Mojo Shape type)
    pattern5 = r'shape\s*=\s*(?:Shape|Dim)?\[?(\d+),\s*(\d+)\]?\s+(?:gbps|GBPS)\s*=\s*([\d.]+)(?:\s+(?:GB/s|GiB/s))?'

    # Pattern 6: shape = [128, 128] gbps = 85.29... (bracket notation)
    pattern6 = r'shape\s*=\s*\[(\d+),\s*(\d+)\]\s+(?:gbps|GBPS)\s*=\s*([\d.]+)(?:\s+(?:GB/s|GiB/s))?'

    data = []

    # Try pattern 1: shape = (x, y)
    matches1 = re.findall(pattern1, stdout)
    for match in matches1:
        dim1, dim2, gbps = match
        shape_str = f"({dim1}, {dim2})"
        data.append({
            'shape': shape_str,
            'gbps': float(gbps)
        })

    # Try pattern 2: size = (x, y)
    matches2 = re.findall(pattern2, stdout)
    for match in matches2:
        dim1, dim2, gbps = match
        shape_str = f"({dim1}, {dim2})"
        data.append({
            'shape': shape_str,
            'gbps': float(gbps)
        })

    # Try pattern 3: size = x (single value)
    matches3 = re.findall(pattern3, stdout)
    for match in matches3:
        size, gbps = match
        shape_str = f"({size},)"
        data.append({
            'shape': shape_str,
            'gbps': float(gbps)
        })

    # Try pattern 4: size = x, y (without parentheses)
    matches4 = re.findall(pattern4, stdout)
    for match in matches4:
        dim1, dim2, gbps = match
        shape_str = f"({dim1}, {dim2})"
        data.append({
            'shape': shape_str,
            'gbps': float(gbps)
        })

    # Try pattern 5: shape = Shape(x, y) or shape = [x, y] (Mojo types)
    matches5 = re.findall(pattern5, stdout)
    for match in matches5:
        dim1, dim2, gbps = match
        shape_str = f"({dim1}, {dim2})"
        data.append({
            'shape': shape_str,
            'gbps': float(gbps)
        })

    # Try pattern 6: shape = [x, y]
    matches6 = re.findall(pattern6, stdout)
    for match in matches6:
        dim1, dim2, gbps = match
        shape_str = f"({dim1}, {dim2})"
        data.append({
            'shape': shape_str,
            'gbps': float(gbps)
        })

    if not data:
        print("Warning: No gbps data found in output", file=sys.stderr)
        return pd.DataFrame(columns=['shape', 'gbps'])

    return pd.DataFrame(data)


def parse_triton_benchmark_table(stdout: str) -> pd.DataFrame:
    """
    Parse the final triton.testing.perf_report table from stdout.
    Dynamically handles any number of columns, including multi-word column names.

    Args:
        stdout: The stdout string from running the benchmark script

    Returns:
        DataFrame with the benchmark table (size, MAX, Triton, Torch, and any other columns)
    """
    lines = stdout.split('\n')

    # Look for the table in the output
    table_lines = []
    in_table = False
    header_line = None
    header_idx = None

    for i, line in enumerate(lines):
        # Look for header line with size/N and provider names
        # Match lines like: "size MAX Triton Torch" or "N MAX Triton Torch"
        if re.search(r'\s+(?:size|N|M)\s+.*\s+(MAX|Triton|Torch)', line, re.IGNORECASE):
            header_idx = i
            header_line = line
            in_table = True
            continue

        # If we found the header, collect data lines
        if in_table and header_idx is not None:
            # Stop at empty line or non-data line
            if not line.strip():
                break
            # Check if line starts with a number (row index)
            if re.match(r'^\s*\d+\s+', line):
                table_lines.append(line)

    if not table_lines or not header_line:
        return pd.DataFrame()

    # Use first data line to determine number of numeric columns
    first_data_parts = table_lines[0].split()
    # First part is row index, rest are numeric values
    num_values = len(first_data_parts) - 1

    # Parse header - get all text after the row number column
    # Split header by multiple spaces to preserve multi-word column names
    header_clean = re.sub(r'^\s*\d*\s*', '', header_line)  # Remove leading number if present

    # Split by 2+ spaces to handle multi-word columns
    columns_raw = re.split(r'\s{2,}', header_clean.strip())

    # Clean up column names and normalize size column
    columns = []
    for col in columns_raw:
        col = col.strip()
        if col:
            # Normalize dimension columns to 'size'
            if col.lower() in ['n', 'm', 'size']:
                columns.append('size')
            else:
                columns.append(col)

    # If we have fewer column names than values, fall back to simple split
    if len(columns) != num_values:
        header_parts = header_line.split()
        columns = []
        found_size = False
        for part in header_parts:
            # Match size, N, M, or similar dimension indicators
            if part.lower() in ['size', 'n', 'm']:
                found_size = True
                columns.append('size')  # Normalize to 'size'
            elif found_size and not part.isdigit():
                columns.append(part)

        # Limit to actual number of data columns
        columns = columns[:num_values]

    if not columns:
        return pd.DataFrame()

    # Parse the table lines
    data = []
    for line in table_lines:
        # Split by whitespace and extract values
        parts = line.split()
        if len(parts) >= len(columns) + 1:  # +1 for the row index
            try:
                row_data = {}
                # Skip first part (row index), then map to columns
                for i, col_name in enumerate(columns):
                    if i + 1 < len(parts):
                        row_data[col_name] = float(parts[i + 1])
                data.append(row_data)
            except (ValueError, IndexError):
                continue

    return pd.DataFrame(data)


def aggregate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean, std, min, max for each shape.

    Args:
        df: DataFrame with 'shape' and 'gbps' columns

    Returns:
        DataFrame with aggregated statistics
    """
    if df.empty:
        return pd.DataFrame(columns=['shape', 'count', 'mean', 'std', 'min', 'max'])

    stats = df.groupby('shape')['gbps'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).reset_index()

    # Sort by shape dimensions for better readability
    def extract_first_dim(shape_str):
        # Match either (x, y) or (x,) format
        match = re.match(r'\((\d+)', shape_str)
        return int(match.group(1)) if match else 0

    stats['sort_key'] = stats['shape'].apply(extract_first_dim)
    stats = stats.sort_values('sort_key').drop('sort_key', axis=1)

    return stats


def run_benchmark(script_path: str, verbose: bool = False) -> tuple[str, str, int]:
    """
    Run the benchmark script and capture stdout/stderr.
    If verbose, stream output in real-time.

    Args:
        script_path: Path to the Python script to run
        verbose: Whether to print verbose output

    Returns:
        Tuple of (stdout, stderr, return_code)
    """
    cmd = ['python', script_path]

    if verbose:
        print(f"Running: {' '.join(cmd)}")
        print("=" * 80)
        print()

    try:
        if verbose:
            # Stream output in real-time using threading
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )

            stdout_lines = []
            stderr_lines = []

            def read_stdout():
                for line in iter(process.stdout.readline, ''):
                    if line:
                        print(line, end='')
                        sys.stdout.flush()
                        stdout_lines.append(line)

            def read_stderr():
                for line in iter(process.stderr.readline, ''):
                    if line:
                        print(line, end='', file=sys.stderr)
                        sys.stderr.flush()
                        stderr_lines.append(line)

            # Start threads to read stdout and stderr
            stdout_thread = threading.Thread(target=read_stdout)
            stderr_thread = threading.Thread(target=read_stderr)

            stdout_thread.start()
            stderr_thread.start()

            # Wait for process to complete
            returncode = process.wait(timeout=600)

            # Wait for threads to finish reading
            stdout_thread.join()
            stderr_thread.join()

            stdout = ''.join(stdout_lines)
            stderr = ''.join(stderr_lines)

            print()
            print("=" * 80)

            return stdout, stderr, returncode
        else:
            # Capture output silently
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            return result.stdout, result.stderr, result.returncode

    except subprocess.TimeoutExpired:
        print("Error: Script execution timed out after 10 minutes", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running script: {e}", file=sys.stderr)
        sys.exit(1)


def get_gpu_info():
    """Get GPU information using torch."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return gpu_name
        else:
            return "CPU"
    except Exception:
        return "Unknown GPU"


def plot_benchmark_results(df: pd.DataFrame, case_name: str, output_path: Path):
    """
    Plot benchmark results with GPU info.

    Args:
        df: DataFrame with benchmark results (size, MAX, Triton, Torch, etc.)
        case_name: Name of the benchmark case
        output_path: Path to save the plot
    """
    gpu_info = get_gpu_info()

    # Create figure with better styling
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors and styles for common providers with distinct colors
    style_map = {
        'MAX': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o'},  # Blue
        'Triton': {'color': '#2ca02c', 'linestyle': '-', 'marker': 's'},  # Green
        'Torch': {'color': '#d62728', 'linestyle': '-', 'marker': '^'},  # Red
        'Custom MAX': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 'D'},  # Orange
        'Custom MAX Elementwise': {'color': '#9467bd', 'linestyle': '-.', 'marker': 'v'},  # Purple
    }

    # Plot each column (except 'size')
    for col in df.columns:
        if col != 'size':
            # Skip columns with all NaN values
            if df[col].isna().all():
                continue

            style = style_map.get(col, {'color': None, 'linestyle': '-', 'marker': 'o'})

            # Filter out NaN values for plotting
            mask = ~df[col].isna()
            if mask.any():
                ax.plot(df.loc[mask, 'size'], df.loc[mask, col],
                       label=col,
                       linewidth=2.5,
                       markersize=7,
                       alpha=0.8,
                       **style)

    # Formatting
    ax.set_xlabel('Size', fontsize=13, fontweight='bold')
    ax.set_ylabel('Throughput (GB/s)', fontsize=13, fontweight='bold')
    ax.set_title(f'{case_name.replace("-", " ").title()} Performance\nGPU: {gpu_info}',
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.95, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Run benchmark scripts and aggregate performance metrics'
    )
    parser.add_argument(
        '--case',
        type=str,
        required=True,
        help='Test case name (e.g., layer-norm, vector-add)'
    )
    parser.add_argument(
        '--script',
        type=str,
        help='Path to the script to run (auto-detected from case if not specified)'
    )
    parser.add_argument(
        '--save',
        type=str,
        help='Directory path to save results (plot and CSV files in case subfolder)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print full stdout/stderr'
    )

    args = parser.parse_args()

    # Auto-detect script path from case name
    if args.script:
        script_path = args.script
    else:
        # Convert case name to script name (e.g., layer-norm -> layer_norm.py)
        script_name = args.case.replace('-', '_') + '.py'

        # Try to find the script in the same directory as this benchmark script
        script_path = Path(__file__).parent / script_name

        # If not found, try in a 'src' subdirectory relative to parent
        if not script_path.exists():
            script_path = Path(__file__).parent.parent / script_name

        if not script_path.exists():
            print(f"Error: Script not found at {script_path}", file=sys.stderr)
            print("Please specify --script manually", file=sys.stderr)
            sys.exit(1)

    # Run the benchmark
    if not args.verbose:
        print(f"Running benchmark for {args.case}... (use --verbose for detailed logs)")

    stdout, stderr, returncode = run_benchmark(str(script_path), verbose=args.verbose)

    # Show stderr on error even if not verbose (verbose mode already streamed it)
    if returncode != 0:
        if not args.verbose and stderr:
            print("\n--- STDERR ---")
            print(stderr)
            print("=" * 80)
        print(f"Warning: Script exited with code {returncode}", file=sys.stderr)

    # Parse the output (check both stdout and stderr)
    if args.verbose:
        print("\nParsing benchmark results...")

    # Try parsing stdout first
    df = parse_gbps_output(stdout)

    # If no data found in stdout, try stderr
    if df.empty and stderr:
        if args.verbose:
            print("No data in stdout, trying stderr...")
        df = parse_gbps_output(stderr)

    # If still no data, try combined output
    if df.empty:
        if args.verbose:
            print("No data in stdout or stderr separately, trying combined...")
        df = parse_gbps_output(stdout + "\n" + stderr)

    if df.empty:
        print("No benchmark data found. Exiting.")
        sys.exit(1)

    if args.verbose:
        print(f"Found {len(df)} benchmark measurements")

    # Aggregate statistics
    stats = aggregate_stats(df)

    # Display aggregated statistics only in verbose mode
    if args.verbose:
        print("\n" + "=" * 80)
        print(f"BENCHMARK RESULTS: {args.case}")
        print("=" * 80)
        print(stats.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
        print("=" * 80)

    # Parse the triton benchmark table and replace MAX column with our computed means
    # The table is typically in stdout, but check both
    triton_table = parse_triton_benchmark_table(stdout)
    if triton_table.empty and stderr:
        triton_table = parse_triton_benchmark_table(stderr)
    if triton_table.empty:
        triton_table = parse_triton_benchmark_table(stdout + "\n" + stderr)

    if not triton_table.empty:
        # Always show the final benchmark table
        print("\n" + "=" * 80)
        print("FINAL BENCHMARK TABLE (GBPS)")
        print("=" * 80)

        # Create mappings from both dimensions to our computed mean
        # This handles cases where either the first or second dimension varies
        first_dim_to_mean = {}
        second_dim_to_mean = {}

        for _, row in stats.iterrows():
            # Extract dimensions from shape string like "(4096, 1024)", "(128,)", or "(128)"
            # Handle formats: (x), (x,), (x, y)
            match = re.match(r'\((\d+)(?:,\s*(\d+)?)?\)', row['shape'])
            if match:
                first_dim = int(match.group(1))
                second_dim = match.group(2)

                first_dim_to_mean[first_dim] = row['mean']
                if second_dim:
                    second_dim_to_mean[int(second_dim)] = row['mean']

        # Replace MAX column with our computed means (custom ops results)
        if 'MAX' in triton_table.columns and (first_dim_to_mean or second_dim_to_mean):
            # Try to map using first dimension
            triton_table['MAX'] = triton_table['size'].map(first_dim_to_mean)

            # If we have NaN values, try second dimension mapping
            if triton_table['MAX'].isna().any() and second_dim_to_mean:
                triton_table['MAX'] = triton_table['MAX'].fillna(
                    triton_table['size'].map(second_dim_to_mean)
                )

        # Convert size to integer
        triton_table['size'] = triton_table['size'].astype(int)

        # Format and display - arrange columns in specific order
        # Put 'size' first, then 'MAX' if it exists, then all other columns
        display_cols = ['size']
        if 'MAX' in triton_table.columns:
            display_cols.append('MAX')

        # Add all other columns in the order they appear
        for col in triton_table.columns:
            if col not in display_cols:
                display_cols.append(col)

        triton_table = triton_table[display_cols]

        # Display the table
        print(triton_table.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
        print("=" * 80)

        # Save results if requested
        if args.save:
            # Create output directory structure
            save_dir = Path(args.save)
            case_dir = save_dir / args.case
            case_dir.mkdir(parents=True, exist_ok=True)

            # Save the benchmark table CSV
            table_csv_path = case_dir / f"{args.case}_benchmark_table.csv"
            triton_table.to_csv(table_csv_path, index=False)

            # Save aggregated stats CSV
            stats_csv_path = case_dir / f"{args.case}_aggregated_stats.csv"
            stats.to_csv(stats_csv_path, index=False)

            # Generate and save plot
            plot_path = case_dir / f"{args.case}_performance.png"
            plot_benchmark_results(triton_table, args.case, plot_path)

            print(f"\n" + "=" * 80)
            print(f"RESULTS SAVED TO: {case_dir}")
            print("=" * 80)
            print(f"  Plot: {plot_path.name}")
            print(f"  Benchmark table: {table_csv_path.name}")
            print(f"  Aggregated stats: {stats_csv_path.name}")
            print("=" * 80)

    # Save just stats to CSV if no triton table but save requested
    elif args.save:
        # Create output directory structure
        save_dir = Path(args.save)
        case_dir = save_dir / args.case
        case_dir.mkdir(parents=True, exist_ok=True)

        # Save aggregated stats CSV
        stats_csv_path = case_dir / f"{args.case}_aggregated_stats.csv"
        stats.to_csv(stats_csv_path, index=False)

        print(f"\n" + "=" * 80)
        print(f"RESULTS SAVED TO: {case_dir}")
        print("=" * 80)
        print(f"  Aggregated stats: {stats_csv_path.name}")
        print("=" * 80)


if __name__ == '__main__':
    main()

