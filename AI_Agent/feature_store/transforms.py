from typing import Optional

def age_band(age: Optional[int]) -> Optional[str]:
    if age is None: return None
    if age < 25: return "18-24"
    if age <= 34: return "25-34"
    if age <= 44: return "35-44"
    if age <= 54: return "45-54"
    if age <= 64: return "55-64"
    return "65+"
