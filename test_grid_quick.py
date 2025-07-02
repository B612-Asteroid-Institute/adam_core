#!/usr/bin/env python3
"""
Quick test version of the grid testing script.
Tests just a few combinations to verify everything works before running the full grid.
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

class QuickGridTester:
    def __init__(self):
        self.results: List[TestResult] = []
        
        # Just test a couple combinations for verification
        self.python_versions = ["3.11"]  # Just one Python version
        self.rebound_versions = ["4.0.2", "4.4.10"]  # Just two rebound versions
        
        # Same tests as the full version
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
                timeout=600,  # 10 minute timeout for quick test
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
                error_message="Test timed out after 10 minutes",
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
    
    def run_quick_tests(self):
        """Run a quick subset of tests"""
        print("🚀 Starting QUICK grid testing...")
        print(f"📊 Testing {len(self.python_versions)} Python versions × {len(self.rebound_versions)} rebound versions × {len(self.test_cases)} tests")
        print(f"📊 Total combinations: {len(self.python_versions) * len(self.rebound_versions) * len(self.test_cases)}")
        print("⚡ This is a quick test with limited combinations to verify the setup works.")
        
        # Backup original pyproject.toml
        self.backup_pyproject()
        
        total_combinations = len(self.python_versions) * len(self.rebound_versions)
        current_combination = 0
        
        try:
            for python_version in self.python_versions:
                for rebound_version in self.rebound_versions:
                    current_combination += 1
                    print(f"\n{'='*80}")
                    print(f"🔄 Quick Test Combination {current_combination}/{total_combinations}")
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
        """Generate a quick test report"""
        print(f"\n{'='*80}")
        print("📋 QUICK TEST REPORT")
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
        
        # Results
        print(f"\n📋 DETAILED RESULTS:")
        for result in self.results:
            status_emoji = {
                "PASSED": "✅",
                "FAILED": "❌",
                "ERROR": "💥", 
                "SKIPPED": "⏭️"
            }
            print(f"   {status_emoji.get(result.status, '❓')} Python {result.python_version} + rebound {result.rebound_version} + {result.test_name}: {result.status}")
            if result.error_message:
                print(f"      💬 {result.error_message}")
        
        if passed == total_tests:
            print(f"\n🎉 All quick tests PASSED! The grid testing script appears to be working correctly.")
            print(f"💡 You can now run the full grid test with: python test_grid.py")
        else:
            print(f"\n⚠️  Some quick tests failed. Please review the errors before running the full grid.")
            print(f"🔧 Fix any issues and try the quick test again.")

def main():
    """Main entry point"""
    print("🔬 Adam Core Quick Grid Tester")
    print("Testing a few combinations to verify the grid testing setup works")
    
    tester = QuickGridTester()
    
    try:
        tester.run_quick_tests()
        tester.generate_report()
        
        # Exit with appropriate code
        failed_count = len([r for r in tester.results if r.status in ["FAILED", "ERROR"]])
        if failed_count > 0:
            print(f"\n⚠️  {failed_count} quick test configurations failed!")
            sys.exit(1)
        else:
            print(f"\n🎉 All quick test configurations passed!")
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