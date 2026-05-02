#!/usr/bin/env python
"""
Quick test of ai_core functionality without needing a GUI or camera.
"""

import sys
import json
from pathlib import Path

# Test 1: Check cache file creation
print("Test 1: Cache system...")
from ai_core import _load_cache, _save_cache, get_cached_profile, set_cached_profile

# Save a test profile
test_profile = {"test": "data", "scraped_at": 123456}
set_cached_profile("test::key", test_profile)

# Load it back
loaded = get_cached_profile("test::key")
assert loaded == test_profile, f"Cache failed: {loaded}"
print("  [OK] Cache system works")

# Test 2: Check payload builder
print("\nTest 2: Payload builder...")
from ai_core import build_payload

test_state = {"emotion": "happy", "metrics": {"fwhr": 1.5}}
payload = build_payload(None, test_state, "github::user1", "linkedin::user2", "resume::/path/file.pdf")
assert payload["face_state"] == test_state, "Payload face_state mismatch"
assert payload["image_b64"] is None, "Payload image should be None"
print("  ✓ Payload builder works")

# Test 3: Check system prompts are defined
print("\nTest 3: System prompts...")
from ai_core import SYSTEM_PROMPTS

for mode in ["glaze", "hate", "super_hate"]:
    assert mode in SYSTEM_PROMPTS, f"Missing system prompt for mode: {mode}"
    assert len(SYSTEM_PROMPTS[mode]) > 0, f"Empty system prompt for {mode}"
print(f"  [OK] All {len(SYSTEM_PROMPTS)} system prompts defined")

# Test 4: Check resume text extraction functions exist
print("\nTest 4: Resume text extraction...")
from ai_core import _extract_text_from_resume

# Test with non-existent file
text = _extract_text_from_resume("/nonexistent/file.pdf")
assert text == "", f"Should return empty string for missing file, got: {text}"
print("  [OK] Resume extraction handles missing files gracefully")

print("\n" + "="*60)
print("All tests passed! [OK]")
print("="*60)
print("\nNext steps:")
print("1. Set environment variables in .env:")
print("   - GROK_API_KEY (from https://console.x.ai)")
print("   - ELEVENLABS_API_KEY")
print("2. Run: python app.py")
print("3. Keybinds:")
print("   - Tab: Trigger AI roast/glaze")
print("   - r: Toggle screen recording")
print("   - space: Capture screenshot")
print("   - m: Toggle mesh visualization")
print("   - esc: Quit")
