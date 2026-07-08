import random
from typing import List, Optional

class SortingAssistant:
    def __init__(self):
        pass

    def run_sorting_step(self, rules: List) -> Optional[dict]:
        """Simulate a camera sorting frame classification against configured rules."""
        if not rules:
            return None
            
        chosen = random.choice(rules)
        result_message = f"Detected Class: '{chosen.object_name}'. Executing rule: IF {chosen.object_name} THEN route to '{chosen.bin_name}'."
        
        return {
            "detected_class": chosen.object_name,
            "bin_name": chosen.bin_name,
            "result_message": result_message
        }
