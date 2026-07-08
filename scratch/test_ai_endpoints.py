import os
import sys
from fastapi.testclient import TestClient

# Add application root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set environment variables for testing
db_file = "test.db"
if os.path.exists(db_file):
    os.remove(db_file)

os.environ["DATABASE_URL"] = f"sqlite:///{db_file}"

from app.main import app
from app.core.db import engine, Base

# Import models to register them with Base.metadata before creating tables
from app.models import models
Base.metadata.create_all(bind=engine)

client = TestClient(app)

def test_gesture_endpoints():
    print("--- Testing Gesture Endpoints ---")
    
    # 1. Get settings
    r = client.get("/api/v1/ai/gesture/settings")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "mappings" in data
    assert "control_enabled" in data
    assert "safety_enabled" in data
    print("Get settings: PASS")

    # 2. Update settings
    r = client.post("/api/v1/ai/gesture/settings", json={"control_enabled": True, "safety_enabled": False})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["control_enabled"] is True
    assert data["safety_enabled"] is False
    print("Update settings: PASS")

    # 3. Create mapping
    r = client.post("/api/v1/ai/gesture/mappings", json={"gesture_name": "Wave", "action": "MOVE_HOME"})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["mappings"]["Wave"] == "MOVE_HOME"
    print("Create mapping: PASS")

    # 4. Delete mapping
    r = client.delete("/api/v1/ai/gesture/mappings/Wave")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "Wave" not in data["mappings"]
    print("Delete mapping: PASS")

    # 5. Train wizard
    r = client.post("/api/v1/ai/gesture/train")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "initiated" in data["message"]
    print("Train wizard: PASS")


def test_voice_endpoints():
    print("--- Testing Voice Endpoints ---")

    # 1. Get settings
    r = client.get("/api/v1/ai/voice/settings")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "commands" in data
    assert "wake_word" in data
    print("Get settings: PASS")

    # 2. Update settings
    r = client.post("/api/v1/ai/voice/settings", json={"wake_word": "Hey Robot", "language": "English (US)"})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["wake_word"] == "Hey Robot"
    assert data["language"] == "English (US)"
    print("Update settings: PASS")

    # 3. Create command rule
    r = client.post("/api/v1/ai/voice/commands", json={"phrase": "go home", "action": "MOVE_HOME", "target": "any"})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    commands = data["commands"]
    assert any(c["phrase"] == "go home" for c in commands)
    rule_id = [c["id"] for c in commands if c["phrase"] == "go home"][0]
    print("Create command rule: PASS")

    # 4. Test command simulator
    r = client.post("/api/v1/ai/voice/test")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "phrase" in data
    assert "intent" in data
    print("Test command simulator: PASS")

    # 5. Delete command rule
    r = client.delete(f"/api/v1/ai/voice/commands/{rule_id}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert not any(c["phrase"] == "go home" for c in data["commands"])
    print("Delete command rule: PASS")


def test_pick_place_endpoints():
    print("--- Testing Pick & Place Endpoints ---")

    # 1. Get settings
    r = client.get("/api/v1/ai/pick-place/settings")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "settings" in data
    assert "grasp_forces" in data
    print("Get settings: PASS")

    # 2. Update settings
    payload = {
        "selection_rule": "Nearest Center",
        "pick_strategy": "Angular Side Reach",
        "pick_min_x": 10.0,
        "pick_max_x": 120.0,
        "drop_min_x": 200.0,
        "drop_max_x": 350.0,
        "grasp_forces": {
            "Fragile Bottle": "Low",
            "Heavy Box": "High"
        }
    }
    r = client.post("/api/v1/ai/pick-place/settings", json=payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["settings"]["selection_rule"] == "Nearest Center"
    assert data["settings"]["pick_strategy"] == "Angular Side Reach"
    assert data["settings"]["pick_min_x"] == 10.0
    assert data["settings"]["pick_max_x"] == 120.0
    assert data["grasp_forces"]["Fragile Bottle"] == "Low"
    assert data["grasp_forces"]["Heavy Box"] == "High"
    print("Update settings: PASS")

    # 3. Simulate path coordinates
    r = client.post("/api/v1/ai/pick-place/simulate")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "path_coords" in data
    assert "log_entries" in data
    assert len(data["path_coords"]) > 0
    print("Simulate path coordinates: PASS")


def test_sorting_endpoints():
    print("--- Testing Smart Sorting Endpoints ---")

    # 1. Get settings
    r = client.get("/api/v1/ai/sorting/settings")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "categories" in data
    assert "rules" in data
    print("Get settings: PASS")

    # 2. Add category bin
    r = client.post("/api/v1/ai/sorting/categories", json={"name": "Glass Bin"})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "Glass bin" in data["categories"]
    print("Add category bin: PASS")

    # 3. Add sorting rule
    r = client.post("/api/v1/ai/sorting/rules", json={"object_name": "Cup", "bin_name": "Glass Bin"})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    rules = data["rules"]
    assert any(rule["object"] == "Cup" and rule["bin"] == "Glass bin" for rule in rules)
    rule_id = [rule["id"] for rule in rules if rule["object"] == "Cup"][0]
    print("Add sorting rule: PASS")

    # 4. Simulate sorting snapshot
    r = client.post("/api/v1/ai/sorting/simulate")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "detected_class" in data
    assert "bin_name" in data
    assert "result_message" in data
    print("Simulate sorting snapshot: PASS")

    # 5. Delete sorting rule
    r = client.delete(f"/api/v1/ai/sorting/rules/{rule_id}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert not any(rule["object"] == "Cup" for rule in data["rules"])
    print("Delete sorting rule: PASS")

    # 6. Delete category bin
    r = client.delete("/api/v1/ai/sorting/categories/Glass bin")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "Glass bin" not in data["categories"]
    print("Delete category bin: PASS")


if __name__ == "__main__":
    try:
        test_gesture_endpoints()
        test_voice_endpoints()
        test_pick_place_endpoints()
        test_sorting_endpoints()
        print("All tests completed successfully!")
    finally:
        # Clean up temporary DB file
        if os.path.exists(db_file):
            os.remove(db_file)
