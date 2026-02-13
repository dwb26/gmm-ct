"""
Verify that the rotation formula in publication_plots.py uses 2pi*omega*t
"""

import numpy as np
from pathlib import Path

# Read the publication.py file
plots_file = Path(__file__).parent.parent.parent / 'gmm_ct' / 'visualization' / 'publication.py'
with open(plots_file, 'r') as f:
    content = f.read()

# Check for the rotation formulas
print("Checking rotation formulas in publication_plots.py...\n")

# Find all instances of theta_t assignments
import re
matches = re.finditer(r'theta_t\s*=\s*([^\n]+)', content)

found_correct = 0
found_incorrect = 0

for match in matches:
    formula = match.group(1).strip()
    line_num = content[:match.start()].count('\n') + 1
    
    if '2 * np.pi' in formula or '2*np.pi' in formula:
        print(f"✓ Line {line_num}: {formula}")
        found_correct += 1
    else:
        print(f"✗ Line {line_num}: {formula} [MISSING 2π FACTOR]")
        found_incorrect += 1

print(f"\nSummary:")
print(f"  Correct formulas: {found_correct}")
print(f"  Incorrect formulas: {found_incorrect}")

if found_incorrect > 0:
    print("\n❌ ERROR: Some rotation formulas are missing the 2π factor!")
    sys.exit(1)
else:
    print("\n✓ All rotation formulas are correct!")
    
# Test the actual numpy calculation
omega = -18.5239
t_val = 0.125
theta_t = 2 * np.pi * omega * t_val
print(f"\nTest calculation with ω={omega}, t={t_val}:")
print(f"  θ = 2π·ω·t = {theta_t:.4f} rad = {np.degrees(theta_t):.2f}°")
print(f"  Expected: ~-833.58° (or ~-113.58° after wrapping)")
