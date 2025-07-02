# Grid Testing for adam-core

This script tests adam-core with different Python and rebound versions to ensure compatibility.

## What it tests

- **Python versions**: 3.11, 3.12
- **Rebound versions**: 4.0.2, 4.0.3, 4.1.1, 4.2.0, 4.3.0, 4.3.1, 4.3.2, 4.4.0, 4.4.1, 4.4.2, 4.4.3, 4.4.5, 4.4.6, 4.4.7, 4.4.8, 4.4.10
- **Specific tests**:
  - `test_generate_impact_visualization_data` from `test_impact_viz_data.py`
  - `test_dinkinesh_propagation` from `test_lambert.py`

**Total combinations**: 2 Python versions × 16 rebound versions × 2 tests = **64 test runs**

## How it works

For each combination, the script:

1. 🧹 Cleans the virtual environment
2. 📝 Updates `pyproject.toml` with pinned rebound version
3. 🔧 Creates new virtual environment with specific Python version
4. 🔒 Runs `pdm lock -G test` to generate lock file
5. 📦 Runs `pdm sync` to install dependencies
6. 🧪 Runs the specific tests using `pdm run pytest`
7. 📊 Records results and timing

## Usage

### Prerequisites

- Python 3.11 and 3.12 installed on your system
- PDM installed and available in PATH
- All system dependencies for adam-core

### Running the tests

```bash
# Run all tests (will take a long time!)
python test_grid.py

# Or make it executable and run directly
./test_grid.py
```

### Expected runtime

This will take a **very long time** to complete:
- ~2-5 minutes per environment setup
- ~1-10 minutes per test (depending on system and test complexity)
- Total estimated time: **2-8 hours**

### Monitoring progress

The script provides real-time updates:
- Shows current combination being tested
- Displays immediate test results
- Provides progress indicators

### Interrupting

You can safely interrupt with `Ctrl+C`. The script will:
- Restore the original `pyproject.toml`
- Clean up temporary environments
- Exit gracefully

## Output

### Console output
- Real-time progress updates
- Immediate test results
- Final summary report

### Files created
- `test_grid_report.json` - Detailed JSON report with all results
- Temporary modifications to `pyproject.toml` (restored after completion)

### Report sections
1. **Summary statistics** - Pass/fail counts and percentages
2. **Detailed results** - Per-configuration breakdown
3. **Failed configurations** - List of problematic combinations
4. **Working configurations** - List of successful combinations

## Example output

```
🔬 Adam Core Grid Tester
Testing Python 3.11 & 3.12 with rebound versions 4.0.2 to 4.4.10
🚀 Starting grid testing...
📊 Testing 2 Python versions × 16 rebound versions × 2 tests
📊 Total combinations: 64

================================================================================
🔄 Combination 1/32
Python 3.11 + rebound 4.0.2
================================================================================
🧹 Cleaning virtual environment...
🔧 Setting up environment: Python 3.11, rebound 4.0.2
🔒 Generating lock file...
📦 Syncing dependencies...
🧪 Running test_impact_viz_data with Python 3.11, rebound 4.0.2
  ✅ test_impact_viz_data: PASSED (45.2s)
🧪 Running test_dinkinesh_propagation with Python 3.11, rebound 4.0.2
  ✅ test_dinkinesh_propagation: PASSED (23.1s)
...
```

## Tips for success

1. **Free up disk space** - Each environment needs ~1-2GB
2. **Ensure stable internet** - Downloads many packages
3. **Run overnight** - This takes many hours
4. **Monitor system resources** - May use significant CPU/memory
5. **Keep system awake** - Prevent sleep during long runs

## Troubleshooting

### Common issues

1. **Python versions not found**
   - Install Python 3.11 and 3.12 via pyenv, asdf, or system package manager
   - Ensure they're in PATH as `python3.11` and `python3.12`

2. **PDM not found**
   - Install PDM: `pip install pdm` or `curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python3 -`

3. **Environment setup failures**
   - Check available disk space
   - Verify internet connection
   - Check Python version compatibility

4. **Test timeouts**
   - Tests have 20-minute timeout
   - Some combinations may be incompatible and hang

### Getting help

If you encounter issues:
1. Check the console output for specific error messages
2. Look at the JSON report for detailed failure information
3. Try running individual combinations manually to debug
4. Check system resources (disk space, memory, CPU)

## Customization

You can modify the script to:
- Test different Python versions
- Test different rebound version ranges
- Add more tests
- Adjust timeouts
- Change reporting format

See the `GridTester` class in `test_grid.py` for configuration options. 