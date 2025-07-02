#!/usr/bin/env python3
"""
Grid testing script for adam-core with different Python and rebound versions.
Tests specific test functions across Python 3.11 and 3.12 with rebound versions 4.0.2 to 4.4.10.
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import tempfile
import shutil

@dataclass
class TestResult:
    python_version: str
    rebound_version: str
    test_name: str
    status: str  # "PASSED", "FAILED", "ERROR", "SKIPPED"
    duration: float
    error_message: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None

class GridTester:
    def __init__(self):
        self.results: List[TestResult] = []
        
        # Python versions to test
        self.python_versions = ["3.11", "3.12"]
        
        # Rebound versions to test (4.0.2 to 4.4.10)
        self.rebound_versions = [
            "4.0.2", "4.0.3", "4.1.1", "4.2.0", "4.3.0", "4.3.1", "4.3.2",
            "4.4.0", "4.4.1", "4.4.2", "4.4.3", "4.4.5", "4.4.6", "4.4.7", "4.4.8", "4.4.10"
        ]
        
        # Specific tests to run
        self.test_cases = [
            {
                "name": "test_impact_viz_data",
                "path": "src/adam_core/dynamics/tests/test_impact_viz_data.py::test_generate_impact_visualization_data"
            },
            {
                "name": "test_dinkinesh_propagation", 
                "path": "src/adam_core/dynamics/tests/test_lambert.py::test_dinkinesh_propagation"
            }
        ]
        
        self.project_root = Path.cwd()
        self.original_pyproject_content = None
        
    def backup_pyproject(self):
        """Backup the original pyproject.toml"""
        pyproject_path = self.project_root / "pyproject.toml"
        if pyproject_path.exists():
            self.original_pyproject_content = pyproject_path.read_text()
        
    def restore_pyproject(self):
        """Restore the original pyproject.toml"""
        if self.original_pyproject_content:
            pyproject_path = self.project_root / "pyproject.toml"
            pyproject_path.write_text(self.original_pyproject_content)
            
    def update_pyproject_rebound(self, rebound_version: str):
        """Update pyproject.toml to pin rebound version"""
        pyproject_path = self.project_root / "pyproject.toml"
        content = pyproject_path.read_text()
        
        # Add rebound dependency if not present, or update if present
        lines = content.split('\n')
        dependencies_section = False
        rebound_added = False
        
        new_lines = []
        for line in lines:
            if line.strip() == 'dependencies = [':
                dependencies_section = True
                new_lines.append(line)
            elif dependencies_section and line.strip() == ']':
                if not rebound_added:
                    new_lines.append(f'  "rebound=={rebound_version}",')
                    rebound_added = True
                new_lines.append(line)
                dependencies_section = False
            elif dependencies_section and 'rebound' in line:
                # Replace existing rebound line
                new_lines.append(f'  "rebound=={rebound_version}",')
                rebound_added = True
            else:
                new_lines.append(line)
        
        # If we never found dependencies section, add it
        if not rebound_added:
            # Find the end of dependencies and add rebound
            content_with_rebound = '\n'.join(new_lines)
            # This is a fallback - the dependencies section should exist
            content_with_rebound = content_with_rebound.replace(
                'dependencies = [',
                f'dependencies = [\n  "rebound=={rebound_version}",'
            )
            new_lines = content_with_rebound.split('\n')
        
        pyproject_path.write_text('\n'.join(new_lines))
        
    def clean_environment(self):
        """Clean PDM virtual environment"""
        print("🧹 Cleaning virtual environment...")
        try:
            # Remove existing venv
            subprocess.run(["pdm", "venv", "remove", "--yes", "in-project"], 
                         capture_output=True, check=False)
            
            # Remove pdm.lock if it exists
            lock_file = self.project_root / "pdm.lock"
            if lock_file.exists():
                lock_file.unlink()
                
            # Remove .venv directory if it exists
            venv_dir = self.project_root / ".venv"
            if venv_dir.exists():
                shutil.rmtree(venv_dir)
                
        except Exception as e:
            print(f"⚠️  Warning: Error cleaning environment: {e}")
            
    def setup_environment(self, python_version: str, rebound_version: str):
        """Setup PDM environment with specific Python and rebound versions"""
        print(f"🔧 Setting up environment: Python {python_version}, rebound {rebound_version}")
        
        try:
            # Update pyproject.toml with rebound version
            self.update_pyproject_rebound(rebound_version)
            
            # Create new virtual environment with specific Python version
            result = subprocess.run([
                "pdm", "venv", "create", "--name", "in-project", f"python{python_version}"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create venv: {result.stderr}")
            
            # Generate lock file
            print("🔒 Generating lock file...")
            result = subprocess.run([
                "pdm", "lock", "-G", "test"
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to generate lock: {result.stderr}")
            
            # Sync dependencies
            print("📦 Syncing dependencies...")
            result = subprocess.run([
                "pdm", "sync", "-G", "test"
            ], capture_output=True, text=True, timeout=900)
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to sync dependencies: {result.stderr}")
                
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup environment: {e}")
            return False
    
    def run_test(self, python_version: str, rebound_version: str, test_case: Dict[str, str]) -> TestResult:
        """Run a specific test case"""
        print(f"🧪 Running {test_case['name']} with Python {python_version}, rebound {rebound_version}")
        
        start_time = time.time()
        
        try:
            # Run the specific test using pdm
            cmd = ["pdm", "run", "pytest", "-v", test_case["path"]]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1200,  # 20 minute timeout per test
                cwd=self.project_root
            )
            
            duration = time.time() - start_time
            
            # Determine test status
            if result.returncode == 0:
                status = "PASSED"
                error_message = None
            elif "SKIPPED" in result.stdout or "SKIPPED" in result.stderr:
                status = "SKIPPED"
                error_message = "Test was skipped"
            else:
                status = "FAILED"
                error_message = f"Exit code: {result.returncode}"
            
            return TestResult(
                python_version=python_version,
                rebound_version=rebound_version,
                test_name=test_case["name"],
                status=status,
                duration=duration,
                error_message=error_message,
                stdout=result.stdout,
                stderr=result.stderr
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                python_version=python_version,
                rebound_version=rebound_version,
                test_name=test_case["name"],
                status="ERROR",
                duration=duration,
                error_message="Test timed out after 20 minutes",
                stdout="",
                stderr=""
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                python_version=python_version,
                rebound_version=rebound_version,
                test_name=test_case["name"],
                status="ERROR",
                duration=duration,
                error_message=str(e),
                stdout="",
                stderr=""
            )
    
    def run_grid_tests(self):
        """Run the complete grid of tests"""
        print("🚀 Starting grid testing...")
        print(f"📊 Testing {len(self.python_versions)} Python versions × {len(self.rebound_versions)} rebound versions × {len(self.test_cases)} tests")
        print(f"📊 Total combinations: {len(self.python_versions) * len(self.rebound_versions) * len(self.test_cases)}")
        
        # Backup original pyproject.toml
        self.backup_pyproject()
        
        total_combinations = len(self.python_versions) * len(self.rebound_versions)
        current_combination = 0
        
        try:
            for python_version in self.python_versions:
                for rebound_version in self.rebound_versions:
                    current_combination += 1
                    print(f"\n{'='*80}")
                    print(f"🔄 Combination {current_combination}/{total_combinations}")
                    print(f"Python {python_version} + rebound {rebound_version}")
                    print(f"{'='*80}")
                    
                    # Clean environment
                    self.clean_environment()
                    
                    # Setup environment
                    if not self.setup_environment(python_version, rebound_version):
                        # If setup failed, mark all tests as errors
                        for test_case in self.test_cases:
                            self.results.append(TestResult(
                                python_version=python_version,
                                rebound_version=rebound_version,
                                test_name=test_case["name"],
                                status="ERROR",
                                duration=0.0,
                                error_message="Environment setup failed"
                            ))
                        continue
                    
                    # Run each test
                    for test_case in self.test_cases:
                        result = self.run_test(python_version, rebound_version, test_case)
                        self.results.append(result)
                        
                        # Print immediate result
                        status_emoji = {
                            "PASSED": "✅",
                            "FAILED": "❌", 
                            "ERROR": "💥",
                            "SKIPPED": "⏭️"
                        }
                        print(f"  {status_emoji.get(result.status, '❓')} {result.test_name}: {result.status} ({result.duration:.1f}s)")
                        if result.error_message:
                            print(f"     Error: {result.error_message}")
        
        finally:
            # Restore original pyproject.toml
            print("\n🔄 Restoring original pyproject.toml...")
            self.restore_pyproject()
            
            # Final cleanup
            print("🧹 Final cleanup...")
            self.clean_environment()
    
    def generate_report(self):
        """Generate a comprehensive test report"""
        print(f"\n{'='*80}")
        print("📋 TEST GRID REPORT")
        print(f"{'='*80}")
        
        # Summary statistics
        total_tests = len(self.results)
        passed = len([r for r in self.results if r.status == "PASSED"])
        failed = len([r for r in self.results if r.status == "FAILED"])
        errors = len([r for r in self.results if r.status == "ERROR"])
        skipped = len([r for r in self.results if r.status == "SKIPPED"])
        
        print(f"📊 SUMMARY:")
        print(f"   Total tests: {total_tests}")
        print(f"   ✅ Passed: {passed} ({passed/total_tests*100:.1f}%)")
        print(f"   ❌ Failed: {failed} ({failed/total_tests*100:.1f}%)")
        print(f"   💥 Errors: {errors} ({errors/total_tests*100:.1f}%)")
        print(f"   ⏭️  Skipped: {skipped} ({skipped/total_tests*100:.1f}%)")
        
        # Detailed results by configuration
        print(f"\n📋 DETAILED RESULTS:")
        for python_version in self.python_versions:
            for rebound_version in self.rebound_versions:
                config_results = [r for r in self.results 
                                if r.python_version == python_version and r.rebound_version == rebound_version]
                
                if config_results:
                    print(f"\n🐍 Python {python_version} + rebound {rebound_version}:")
                    for result in config_results:
                        status_emoji = {
                            "PASSED": "✅",
                            "FAILED": "❌",
                            "ERROR": "💥", 
                            "SKIPPED": "⏭️"
                        }
                        print(f"   {status_emoji.get(result.status, '❓')} {result.test_name}: {result.status} ({result.duration:.1f}s)")
                        if result.error_message:
                            print(f"      💬 {result.error_message}")
        
        # Failed configurations
        failed_results = [r for r in self.results if r.status in ["FAILED", "ERROR"]]
        if failed_results:
            print(f"\n❌ FAILED CONFIGURATIONS:")
            for result in failed_results:
                print(f"   Python {result.python_version} + rebound {result.rebound_version} + {result.test_name}")
                if result.error_message:
                    print(f"      💬 {result.error_message}")
                if result.stderr and result.stderr.strip():
                    print(f"      📝 stderr: {result.stderr.strip()[:200]}...")
        
        # Working configurations
        passed_results = [r for r in self.results if r.status == "PASSED"]
        if passed_results:
            print(f"\n✅ WORKING CONFIGURATIONS:")
            for result in passed_results:
                print(f"   Python {result.python_version} + rebound {result.rebound_version} + {result.test_name}")
        
        # Save detailed JSON report
        report_data = {
            "summary": {
                "total": total_tests,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "skipped": skipped
            },
            "results": [asdict(r) for r in self.results]
        }
        
        report_file = self.project_root / "test_grid_report.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_file}")

def main():
    """Main entry point"""
    print("🔬 Adam Core Grid Tester")
    print("Testing Python 3.11 & 3.12 with rebound versions 4.0.2 to 4.4.10")
    
    tester = GridTester()
    
    try:
        tester.run_grid_tests()
        tester.generate_report()
        
        # Exit with appropriate code
        failed_count = len([r for r in tester.results if r.status in ["FAILED", "ERROR"]])
        if failed_count > 0:
            print(f"\n⚠️  {failed_count} test configurations failed!")
            sys.exit(1)
        else:
            print(f"\n🎉 All test configurations passed!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
        tester.restore_pyproject()
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        tester.restore_pyproject()
        sys.exit(1)

if __name__ == "__main__":
    main() 