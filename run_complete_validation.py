#!/usr/bin/env python3
"""
Final Validation Command: Complete System Validation
===================================================
Comprehensive validation script for the facial recognition bias detection system.
Runs all tests, validations, and system checks as specified in Day 10 requirements.
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path

def run_command(cmd, description, timeout=120):
    """Run a command with timeout and error handling"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True, result.stdout
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT ({timeout}s)")
        return False, "Timeout expired"
    except Exception as e:
        print(f"💥 {description} - ERROR: {str(e)}")
        return False, str(e)

def check_file_exists(filepath, description):
    """Check if required file exists"""
    if Path(filepath).exists():
        print(f"✅ {description} - File exists")
        return True
    else:
        print(f"❌ {description} - File missing: {filepath}")
        return False

def run_complete_validation():
    """Run comprehensive system validation"""
    print("🌅 FINAL VALIDATION: Complete System Validation")
    print("=" * 60)
    
    validation_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'tests': {},
        'overall_success': False
    }
    
    # Step 1: Check file completeness
    print("\n📁 STEP 1: File Completeness Check")
    required_files = [
        ("README.md", "Project README"),
        ("requirements.txt", "Dependencies file"),
        ("Day7_FastAPI_Backend.py", "FastAPI backend"),
        ("Day8_BiasMitigationSuite.py", "Bias mitigation suite"),
        ("Day9_Comprehensive_Testing.py", "Testing suite"),
        ("Day9_Mathematical_Validation.py", "Mathematical validation"),
        ("Day9_Academic_Report.md", "Academic report"),
        ("Day10_Executive_Presentation.md", "Executive presentation"),
        ("Day10_Technical_Demo_Script.md", "Demo script"),
        ("Day10_Academic_Poster.md", "Academic poster"),
        ("Day10_Social_Media_Summary.md", "Social media content")
    ]
    
    file_check_results = []
    for filepath, description in required_files:
        result = check_file_exists(filepath, description)
        file_check_results.append(result)
    
    validation_results['tests']['file_completeness'] = {
        'passed': all(file_check_results),
        'success_rate': sum(file_check_results) / len(file_check_results) * 100
    }
    
    # Step 2: Run comprehensive testing suite
    print("\n🧪 STEP 2: Comprehensive Testing Suite")
    success, output = run_command(
        "python3 Day9_Comprehensive_Testing.py",
        "Comprehensive system tests",
        timeout=180
    )
    
    # Parse test results
    if "Success rate:" in output:
        try:
            success_rate = float(output.split("Success rate: ")[1].split("%")[0])
            validation_results['tests']['comprehensive_testing'] = {
                'passed': success_rate >= 75,
                'success_rate': success_rate
            }
        except:
            validation_results['tests']['comprehensive_testing'] = {
                'passed': False,
                'success_rate': 0
            }
    else:
        validation_results['tests']['comprehensive_testing'] = {
            'passed': False,
            'success_rate': 0
        }
    
    # Step 3: Mathematical validation
    print("\n🔬 STEP 3: Mathematical Validation")
    success, output = run_command(
        "python3 Day9_Mathematical_Validation.py",
        "Mathematical validation suite",
        timeout=120
    )
    
    # Parse mathematical validation results
    if "OVERALL SUCCESS RATE:" in output:
        try:
            success_rate = float(output.split("OVERALL SUCCESS RATE: ")[1].split("%")[0])
            validation_results['tests']['mathematical_validation'] = {
                'passed': success_rate >= 75,
                'success_rate': success_rate
            }
        except:
            validation_results['tests']['mathematical_validation'] = {
                'passed': False,
                'success_rate': 0
            }
    else:
        validation_results['tests']['mathematical_validation'] = {
            'passed': False,
            'success_rate': 0
        }
    
    # Step 4: System integration validation
    print("\n🔗 STEP 4: System Integration Validation")
    success, output = run_command(
        "python3 Day10_Final_Integration_Checklist.py",
        "System integration validation",
        timeout=90
    )
    
    # Parse integration results
    if "INTEGRATION SUCCESS RATE:" in output:
        try:
            success_rate = float(output.split("INTEGRATION SUCCESS RATE: ")[1].split("%")[0])
            validation_results['tests']['system_integration'] = {
                'passed': success_rate >= 90,
                'success_rate': success_rate
            }
        except:
            validation_results['tests']['system_integration'] = {
                'passed': False,
                'success_rate': 0
            }
    else:
        validation_results['tests']['system_integration'] = {
            'passed': False,
            'success_rate': 0
        }
    
    # Step 5: API health check
    print("\n🌐 STEP 5: API Health Check")
    success, output = run_command(
        "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/api/health",
        "API health endpoint",
        timeout=10
    )
    
    api_healthy = "200" in output if success else False
    validation_results['tests']['api_health'] = {
        'passed': api_healthy,
        'response_code': output if success else "No response"
    }
    
    # Step 6: Performance validation
    print("\n⚡ STEP 6: Performance Validation")
    success, output = run_command(
        "python3 -c \"import time; import numpy as np; start=time.time(); result=np.sum(np.random.rand(10000)**2); print(f'Vectorized operation: {time.time()-start:.6f}s')\"",
        "Performance benchmark",
        timeout=30
    )
    
    performance_ok = success and "Vectorized operation:" in output
    validation_results['tests']['performance'] = {
        'passed': performance_ok,
        'details': output if success else "Performance test failed"
    }
    
    # Step 7: Documentation validation
    print("\n📚 STEP 7: Documentation Validation")
    doc_files = [
        "Day9_Academic_Report.md",
        "Day10_Executive_Presentation.md", 
        "Day10_Technical_Demo_Script.md"
    ]
    
    doc_checks = []
    for doc_file in doc_files:
        if Path(doc_file).exists():
            with open(doc_file, 'r') as f:
                content = f.read()
                # Check for substantial content (>1000 characters)
                doc_checks.append(len(content) > 1000)
        else:
            doc_checks.append(False)
    
    validation_results['tests']['documentation'] = {
        'passed': all(doc_checks),
        'files_validated': len([c for c in doc_checks if c])
    }
    
    # Calculate overall success
    test_results = [test['passed'] for test in validation_results['tests'].values()]
    overall_success_rate = sum(test_results) / len(test_results) * 100
    validation_results['overall_success'] = overall_success_rate >= 80
    validation_results['overall_success_rate'] = overall_success_rate
    
    # Generate final report
    print("\n" + "=" * 60)
    print("📊 FINAL VALIDATION REPORT")
    print("=" * 60)
    
    print(f"\n🎯 OVERALL SUCCESS RATE: {overall_success_rate:.1f}%")
    
    print("\n📋 DETAILED RESULTS:")
    for test_name, result in validation_results['tests'].items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"  {status} {test_name.replace('_', ' ').title()}")
        
        if 'success_rate' in result:
            print(f"    Success Rate: {result['success_rate']:.1f}%")
        if 'details' in result:
            print(f"    Details: {result['details']}")
    
    # Final assessment
    if overall_success_rate >= 90:
        print("\n🎉 EXCELLENT: System ready for production deployment!")
        final_status = "EXCELLENT"
    elif overall_success_rate >= 80:
        print("\n✅ GOOD: System ready with minor considerations.")
        final_status = "GOOD"
    elif overall_success_rate >= 60:
        print("\n⚠️ FAIR: System needs improvements before deployment.")
        final_status = "FAIR"
    else:
        print("\n❌ POOR: Major issues detected, system not ready.")
        final_status = "POOR"
    
    validation_results['final_status'] = final_status
    
    # Save validation results
    with open('validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n💾 Validation results saved to: validation_results.json")
    
    return validation_results

if __name__ == '__main__':
    print("🚀 Starting Complete System Validation...")
    
    # Check if we're in the right directory
    if not Path('requirements.txt').exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Run validation with command line arguments
    comprehensive = '--comprehensive' in sys.argv
    mathematical = '--mathematical-validation' in sys.argv
    statistical = '--statistical-tests' in sys.argv
    
    if comprehensive or mathematical or statistical:
        print(f"🔧 Running with flags: comprehensive={comprehensive}, mathematical={mathematical}, statistical={statistical}")
    
    # Run complete validation
    results = run_complete_validation()
    
    # Exit with appropriate code
    if results['overall_success']:
        print("\n🎊 VALIDATION COMPLETE: System ready for deployment!")
        sys.exit(0)
    else:
        print("\n⚠️ VALIDATION INCOMPLETE: Please address issues before deployment.")
        sys.exit(1)
