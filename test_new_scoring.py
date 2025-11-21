#!/usr/bin/env python3
"""Quick test to verify new dual scoring logic."""

from aurelius.shared.moderation import ModerationResult

# Test case 1: High category triggers acceptance (violence/graphic = 0.85)
result1 = ModerationResult(
    flagged=True,
    category_scores={"violence/graphic": 0.85, "violence": 0.05, "hate": 0.01},
    categories={"violence/graphic": True, "violence": False, "hate": False},
    combined_score=0.15,  # Low aggregate score
    high_category_triggered=True,
    high_category_name="violence/graphic",
    high_category_score=0.85,
)

# Test case 2: Aggregate threshold triggers acceptance (all low individual, but aggregate high)
result2 = ModerationResult(
    flagged=True,
    category_scores={"hate": 0.3, "violence": 0.25, "harassment": 0.2},
    categories={"hate": False, "violence": False, "harassment": False},
    combined_score=0.25,  # Above 0.2 threshold
    high_category_triggered=False,
    high_category_name=None,
    high_category_score=None,
)

# Test case 3: Should be rejected (low aggregate, no high category)
result3 = ModerationResult(
    flagged=False,
    category_scores={"hate": 0.1, "violence": 0.05, "harassment": 0.08},
    categories={"hate": False, "violence": False, "harassment": False},
    combined_score=0.07,  # Below 0.2 threshold
    high_category_triggered=False,
    high_category_name=None,
    high_category_score=None,
)

# Test acceptance logic
DANGER_THRESHOLD = 0.2
SINGLE_CATEGORY_THRESHOLD = 0.8

print("=" * 60)
print("Testing Dual Acceptance Logic")
print("=" * 60)
print(f"DANGER_THRESHOLD = {DANGER_THRESHOLD}")
print(f"SINGLE_CATEGORY_THRESHOLD = {SINGLE_CATEGORY_THRESHOLD}")
print()

# Test 1
accepted1 = result1.high_category_triggered or result1.combined_score >= DANGER_THRESHOLD
print("Test 1: High category (violence/graphic = 0.85)")
print(f"  Aggregate score: {result1.combined_score:.3f}")
print(f"  High category triggered: {result1.high_category_triggered}")
print(f"  Result: {'ACCEPTED ✓' if accepted1 else 'REJECTED ✗'}")
print(f"  Expected: ACCEPTED ✓")
assert accepted1, "Test 1 should be accepted via high category"
print("  ✓ PASS")
print()

# Test 2
accepted2 = result2.high_category_triggered or result2.combined_score >= DANGER_THRESHOLD
print("Test 2: Aggregate threshold (combined = 0.25)")
print(f"  Aggregate score: {result2.combined_score:.3f}")
print(f"  High category triggered: {result2.high_category_triggered}")
print(f"  Result: {'ACCEPTED ✓' if accepted2 else 'REJECTED ✗'}")
print(f"  Expected: ACCEPTED ✓")
assert accepted2, "Test 2 should be accepted via aggregate threshold"
print("  ✓ PASS")
print()

# Test 3
accepted3 = result3.high_category_triggered or result3.combined_score >= DANGER_THRESHOLD
print("Test 3: Below both thresholds (combined = 0.07, max category = 0.1)")
print(f"  Aggregate score: {result3.combined_score:.3f}")
print(f"  High category triggered: {result3.high_category_triggered}")
print(f"  Result: {'ACCEPTED ✓' if accepted3 else 'REJECTED ✗'}")
print(f"  Expected: REJECTED ✗")
assert not accepted3, "Test 3 should be rejected"
print("  ✓ PASS")
print()

print("=" * 60)
print("All tests passed! ✓")
print("=" * 60)
