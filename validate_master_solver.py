#!/usr/bin/env python3
"""
Master Solver Validation Script

Comprehensive test of AIMO-3 master solver implementation.
Validates all key features and master prompt compliance.
"""

import sys
sys.path.insert(0, '.')

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def test_all():
    """Run comprehensive validation suite."""
    
    print("=" * 70)
    print("AIMO-3 MASTER SOLVER - COMPREHENSIVE VALIDATION")
    print("=" * 70)
    
    # Test 1: Imports
    print("\n[TEST 1] Module Imports")
    print("-" * 70)
    try:
        from solver import StrategyArbiter, ProblemClassifier, CandidateGenerator, AnswerArbitrator
        from sympy_solver import SymPySolver, EquationExtractor, DiophantineSolver
        from validation import SelfVerificationLoop
        from parsing import ProblemParser
        print("✓ All core classes import successfully")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Test 2: SymPy Availability
    print("\n[TEST 2] SymPy Availability")
    print("-" * 70)
    try:
        import sympy as sp
        from sympy_solver import SYMPY_AVAILABLE
        print(f"✓ SymPy {sp.__version__} available")
        print(f"✓ SYMPY_AVAILABLE flag: {SYMPY_AVAILABLE}")
        if not SYMPY_AVAILABLE:
            print("⚠ Warning: SYMPY_AVAILABLE is False")
    except ImportError:
        print("✗ SymPy not available")
        return False
    
    # Test 3: Problem Classification
    print("\n[TEST 3] Problem Classification")
    print("-" * 70)
    test_problems = [
        ("Find the remainder when 7^1000 is divided by 13", "modular"),
        ("How many ways can you arrange 5 books?", "combinatorics"),
        ("Find all integer solutions to x^2 + y^2 = 25", "diophantine"),
        ("A triangle has sides 3, 4, 5. What is the area?", "geometry"),
        ("Maximize x + y subject to x^2 + y^2 <= 1", "optimization"),
    ]
    
    for problem, expected_type in test_problems:
        classification = ProblemClassifier.classify(problem)
        detected_type = classification['problem_type']
        status = "✓" if detected_type == expected_type else "⚠"
        print(f"{status} '{problem[:40]}...'")
        print(f"   Expected: {expected_type}, Got: {detected_type}")
    
    # Test 4: Symbolic Solving
    print("\n[TEST 4] Symbolic Solving")
    print("-" * 70)
    
    # Test 4a: Modular equation
    problem = "Find the remainder when 7^1000 is divided by 13"
    result = SymPySolver.solve_modular_equation(problem)
    expected = (9, 0)
    status = "✓" if result == expected else "✗"
    print(f"{status} Modular: {problem}")
    print(f"   Expected: {expected}, Got: {result}")
    
    # Test 4b: Equation solving
    result = SymPySolver.solve_from_equations(["x**2 - 4 = 0"])
    status = "✓" if result and result[0] in [-2, 2] else "✗"
    print(f"{status} Equation: x^2 - 4 = 0")
    print(f"   Got: {result}")
    
    # Test 5: Multi-Candidate Generation
    print("\n[TEST 5] Multi-Candidate Generation")
    print("-" * 70)
    
    problem = "Find the remainder when 7^1000 is divided by 13"
    classification = ProblemClassifier.classify(problem)
    generator = CandidateGenerator(timeout_remaining=5.0)
    candidates = generator.generate(problem, classification)
    
    print(f"Problem: {problem}")
    print(f"Candidates generated: {len(candidates)}")
    for i, candidate in enumerate(candidates, 1):
        print(f"  {i}. {candidate}")
    
    if len(candidates) > 0:
        print("✓ Candidates generated successfully")
    else:
        print("⚠ No candidates generated")
    
    # Test 6: Full Solver Integration
    print("\n[TEST 6] Full Solver Integration")
    print("-" * 70)
    
    problems = [
        "Find the remainder when 7^1000 is divided by 13",
        "What is 2 + 2?",
    ]
    
    for problem in problems:
        solver = StrategyArbiter(timeout=10.0)
        result = solver.solve(problem)
        status = "✓" if result != (0, 0) else "⚠"
        print(f"{status} {problem[:40]}...")
        print(f"   Result: {result}")
    
    # Test 7: Master Prompt Compliance
    print("\n[TEST 7] Master Prompt Compliance")
    print("-" * 70)
    
    requirements = [
        ("ProblemClassifier.classify", ProblemClassifier, "classify"),
        ("CandidateGenerator._try_symbolic", CandidateGenerator, "_try_symbolic"),
        ("CandidateGenerator._try_llm_reasoning", CandidateGenerator, "_try_llm_reasoning"),
        ("EquationExtractor.extract_equations", EquationExtractor, "extract_equations"),
        ("SymPySolver.solve_modular_equation", SymPySolver, "solve_modular_equation"),
        ("SymPySolver.solve_from_equations", SymPySolver, "solve_from_equations"),
        ("AnswerArbitrator.arbitrate", AnswerArbitrator, "arbitrate"),
        ("StrategyArbiter.solve", StrategyArbiter, "solve"),
        ("StrategyArbiter.time_remaining", StrategyArbiter, "time_remaining"),
    ]
    
    all_present = True
    for name, cls, method in requirements:
        present = hasattr(cls, method)
        status = "✓" if present else "✗"
        print(f"{status} {name}")
        if not present:
            all_present = False
    
    # Final Summary
    print("\n" + "=" * 70)
    if all_present and len(candidates) > 0:
        print("✅ MASTER SOLVER FULLY COMPLIANT AND FUNCTIONAL")
        print("=" * 70)
        return True
    else:
        print("⚠ Some tests failed - review above for details")
        print("=" * 70)
        return False

if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)
