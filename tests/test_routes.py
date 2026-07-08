from unittest.mock import MagicMock, patch


# ── Health & Root ─────────────────────────────────────────────────────────────

def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]


# ── Tasks ─────────────────────────────────────────────────────────────────────

def test_list_tasks(client):
    response = client.get("/api/v1/ai/tasks")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 6
    task_ids = [t["id"] for t in data]
    assert "obj-detect" in task_ids
    assert "face-rec" in task_ids


def test_start_task_valid(client):
    response = client.post("/api/v1/ai/tasks/start", json={"task_id": "obj-detect"})
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "active"
    assert data["task_id"] == "obj-detect"


def test_start_task_invalid(client):
    response = client.post("/api/v1/ai/tasks/start", json={"task_id": "invalid-task"})
    assert response.status_code == 400
    assert "Invalid task ID" in response.json()["detail"]


def test_stop_task_valid(client):
    # Start first
    client.post("/api/v1/ai/tasks/start", json={"task_id": "face-rec"})
    response = client.post("/api/v1/ai/tasks/stop", json={"task_id": "face-rec"})
    assert response.status_code == 200
    assert response.json()["state"] == "idle"


def test_stop_task_invalid(client):
    response = client.post("/api/v1/ai/tasks/stop", json={"task_id": "bad-task"})
    assert response.status_code == 400


def test_start_all_tasks(client):
    response = client.post("/api/v1/ai/tasks/start-all")
    assert response.status_code == 200
    assert response.json()["state"] == "all_active"


def test_stop_all_tasks(client):
    response = client.post("/api/v1/ai/tasks/stop-all")
    assert response.status_code == 200
    assert response.json()["state"] == "all_idle"


# ── Status ────────────────────────────────────────────────────────────────────

def test_get_status_no_camera(client):
    with patch("app.api.routes.status.check_camera_connected", return_value="disconnected"):
        response = client.get("/api/v1/ai/status")
    assert response.status_code == 200
    data = response.json()
    assert "model_status" in data
    assert "camera_status" in data


def test_get_status_camera_connected(client):
    with patch("app.api.routes.status.check_camera_connected", return_value="connected"):
        response = client.get("/api/v1/ai/status?camera_url=http://localhost:81/stream")
    assert response.status_code == 200
    assert response.json()["camera_status"] == "connected"


def test_check_camera_connected_success():
    from app.api.routes.status import check_camera_connected
    with patch("socket.create_connection") as mock_conn:
        mock_conn.return_value.__enter__ = MagicMock()
        mock_conn.return_value.__exit__ = MagicMock()
        result = check_camera_connected("http://192.168.1.105:81/stream")
        assert result == "connected"


def test_check_camera_connected_failure():
    from app.api.routes.status import check_camera_connected
    with patch("socket.create_connection", side_effect=Exception("Timeout")):
        result = check_camera_connected("http://192.168.1.105:81/stream")
        assert result == "disconnected"


# ── Face ──────────────────────────────────────────────────────────────────────

def test_list_operators_empty(client):
    response = client.get("/api/v1/ai/face/operators")
    assert response.status_code == 200
    assert response.json() == []


def test_register_face_success(client):
    from app.core.state import face_recognizer
    face_recognizer.register_face = MagicMock(return_value="/uploads/operators/john_doe_abc.jpg")
    
    file_bytes = b"fake-jpeg-data"
    response = client.post(
        "/api/v1/ai/face/register",
        files={"files": ("photo.jpg", file_bytes, "image/jpeg")},
        data={"name": "John Doe", "access_level": "admin"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["name"] == "John Doe"
    assert data["access_level"] == "admin"


def test_register_face_no_faces_detected(client):
    from app.core.state import face_recognizer
    face_recognizer.register_face = MagicMock(return_value=None)
    
    response = client.post(
        "/api/v1/ai/face/register",
        files={"files": ("photo.jpg", b"bad-data", "image/jpeg")},
        data={"name": "Nobody", "access_level": "viewer"}
    )
    assert response.status_code == 400
    assert "No valid faces" in response.json()["detail"]


def test_update_operator_success(client, db):
    from app.models import Operator
    op = Operator(name="Alice", face_path="alice_abc.jpg", access_level="viewer")
    db.add(op)
    db.commit()
    db.refresh(op)
    
    from app.core.state import face_recognizer
    face_recognizer.rename_operator = MagicMock()
    
    response = client.put(
        f"/api/v1/ai/face/operators/{op.id}",
        json={"name": "Alice Smith", "access_level": "admin"}
    )
    assert response.status_code == 200


def test_update_operator_not_found(client):
    response = client.put("/api/v1/ai/face/operators/9999", json={"access_level": "admin"})
    assert response.status_code == 404


def test_delete_operator_success(client, db):
    from app.models import Operator
    op = Operator(name="Bob", face_path="bob_abc.jpg", access_level="viewer")
    db.add(op)
    db.commit()
    db.refresh(op)
    
    from app.core.state import face_recognizer
    face_recognizer.remove_operator = MagicMock()
    
    response = client.delete(f"/api/v1/ai/face/operators/{op.id}")
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_delete_operator_not_found(client):
    response = client.delete("/api/v1/ai/face/operators/9999")
    assert response.status_code == 404


def test_retrain_model(client):
    from app.core.state import face_recognizer
    face_recognizer._collection = MagicMock()
    face_recognizer._collection.count.return_value = 5
    
    response = client.post("/api/v1/ai/face/retrain")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["embedding_count"] == 5


def test_retrain_model_error(client):
    from app.core.state import face_recognizer
    face_recognizer._collection = MagicMock()
    face_recognizer._collection.count.side_effect = Exception("ChromaDB Error")
    
    response = client.post("/api/v1/ai/face/retrain")
    assert response.status_code == 500


def test_face_stats(client):
    from app.core.state import face_recognizer
    face_recognizer._collection = MagicMock()
    face_recognizer._collection.count.return_value = 3
    
    response = client.get("/api/v1/ai/face/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["embedding_count"] == 3
    assert data["ready"] is True


def test_face_stats_error(client):
    from app.core.state import face_recognizer
    face_recognizer._collection = MagicMock()
    face_recognizer._collection.count.side_effect = Exception("Collection Error")
    
    response = client.get("/api/v1/ai/face/stats")
    assert response.status_code == 500


# ── Gesture ───────────────────────────────────────────────────────────────────

def test_get_gesture_settings(client):
    response = client.get("/api/v1/ai/gesture/settings")
    assert response.status_code == 200
    data = response.json()
    assert "control_enabled" in data
    assert "safety_enabled" in data


def test_update_gesture_settings(client):
    response = client.post(
        "/api/v1/ai/gesture/settings",
        json={"control_enabled": True, "safety_enabled": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["control_enabled"] is True
    assert data["safety_enabled"] is False


def test_add_gesture_mapping(client):
    response = client.post(
        "/api/v1/ai/gesture/mappings",
        json={"gesture_name": "thumbs_up", "action": "BASE LEFT"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_add_gesture_mapping_missing_name(client):
    response = client.post("/api/v1/ai/gesture/mappings", json={"action": "BASE LEFT"})
    assert response.status_code == 400


def test_add_gesture_mapping_invalid_action(client):
    response = client.post(
        "/api/v1/ai/gesture/mappings",
        json={"gesture_name": "wave", "action": "INVALID_ACTION"}
    )
    assert response.status_code == 422


def test_delete_gesture_mapping_success(client, db):
    from app.models import GestureMapping
    mapping = GestureMapping(gesture_name="test_wave", action="BASE RIGHT")
    db.add(mapping)
    db.commit()
    
    from app.core.state import gesture_controller
    gesture_controller.remove_gesture = MagicMock(return_value=0)
    
    response = client.delete("/api/v1/ai/gesture/mappings/test_wave")
    assert response.status_code == 200


def test_delete_gesture_mapping_not_found(client):
    response = client.delete("/api/v1/ai/gesture/mappings/nonexistent_gesture")
    assert response.status_code == 404


def test_register_custom_gesture(client):
    from app.core.state import gesture_controller
    gesture_controller.register_from_frame = MagicMock(
        return_value={"vector_id": "vec-123", "sample_count": 1}
    )
    
    response = client.post(
        "/api/v1/ai/gesture/custom/register",
        files={"frame": ("frame.jpg", b"fake-frame-data", "image/jpeg")},
        data={"gesture_name": "peace_sign", "action": "OPEN GRIP"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["gesture_name"] == "peace_sign"


def test_register_custom_gesture_error(client):
    from app.core.state import gesture_controller
    gesture_controller.register_from_frame = MagicMock(side_effect=Exception("No hand detected"))
    
    response = client.post(
        "/api/v1/ai/gesture/custom/register",
        files={"frame": ("frame.jpg", b"bad-data", "image/jpeg")},
        data={"gesture_name": "bad_gesture"}
    )
    assert response.status_code == 400
    assert "No hand detected" in response.json()["detail"]


def test_list_custom_gestures(client):
    from app.core.state import gesture_controller
    gesture_controller.list_gestures = MagicMock(return_value={"peace_sign": 3, "thumbs_up": 5})
    
    response = client.get("/api/v1/ai/gesture/custom")
    assert response.status_code == 200
    data = response.json()
    assert data["total_gestures"] == 2


def test_delete_custom_gesture(client):
    from app.core.state import gesture_controller
    gesture_controller.remove_gesture = MagicMock(return_value=3)
    
    response = client.delete("/api/v1/ai/gesture/custom/peace_sign")
    assert response.status_code == 200
    data = response.json()
    assert data["vectors_deleted"] == 3


def test_train_gesture(client):
    from app.core.state import gesture_controller
    gesture_controller.train_wizard = MagicMock(return_value={"status": "ready"})
    
    response = client.post("/api/v1/ai/gesture/train")
    assert response.status_code == 200


# ── Voice ─────────────────────────────────────────────────────────────────────

def test_get_voice_commands(client):
    response = client.get("/api/v1/ai/voice/commands")
    assert response.status_code == 200
    data = response.json()
    assert "wake_word" in data
    assert "commands" in data
    # Default commands should be seeded
    assert len(data["commands"]) == 8


def test_update_voice_settings(client):
    response = client.post(
        "/api/v1/ai/voice/settings",
        json={"wake_word": "Hey Grabber", "language": "Spanish (ES)"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["wake_word"] == "Hey Grabber"
    assert data["language"] == "Spanish (ES)"


# ── Pick & Place ──────────────────────────────────────────────────────────────

def test_get_pick_place_settings(client):
    response = client.get("/api/v1/ai/pick-place/settings")
    assert response.status_code == 200
    data = response.json()
    assert "selection_rule" in data
    assert "workspace_coords" in data
    assert "grasp_forces" in data


def test_update_pick_place_settings(client):
    response = client.post(
        "/api/v1/ai/pick-place/settings",
        json={
            "selection_rule": "Closest First",
            "pick_strategy": "Side Approach",
            "workspace_coords": {"pickMinX": -50.0, "pickMaxX": 50.0}
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["selection_rule"] == "Closest First"


def test_simulate_pick_place(client):
    response = client.post("/api/v1/ai/pick-place/simulate")
    assert response.status_code == 200
    data = response.json()
    assert "path_coords" in data
    assert "log_entries" in data
    assert len(data["path_coords"]) == 121


# ── Sorting ───────────────────────────────────────────────────────────────────

def test_get_sorting_settings(client):
    response = client.get("/api/v1/ai/sorting/settings")
    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert "rules" in data


def test_add_sorting_category(client):
    response = client.post(
        "/api/v1/ai/sorting/categories",
        json={"name": "glass"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "Glass" in data["categories"]


def test_add_sorting_category_missing_name(client):
    response = client.post("/api/v1/ai/sorting/categories", json={})
    assert response.status_code == 400


def test_add_sorting_category_duplicate(client):
    client.post("/api/v1/ai/sorting/categories", json={"name": "plastic"})
    response = client.post("/api/v1/ai/sorting/categories", json={"name": "plastic"})
    assert response.status_code == 200


def test_add_sorting_rule(client, db):
    from app.models import SortingCategory
    # Seed 'Plastic' category (it might already exist from seeding, but explicitly add)
    existing = db.query(SortingCategory).filter(SortingCategory.name == "Plastic").first()
    if not existing:
        db.add(SortingCategory(name="Plastic"))
        db.commit()
    response = client.post(
        "/api/v1/ai/sorting/rules",
        json={"object_name": "bottle", "bin_name": "Plastic"}
    )
    assert response.status_code == 200


def test_add_sorting_rule_missing_fields(client):
    response = client.post("/api/v1/ai/sorting/rules", json={"object_name": "bottle"})
    assert response.status_code == 400


def test_add_sorting_rule_invalid_bin(client):
    response = client.post(
        "/api/v1/ai/sorting/rules",
        json={"object_name": "Widget", "bin_name": "NonExistentBin"}
    )
    assert response.status_code == 400


def test_delete_sorting_category(client, db):
    from app.models import SortingCategory
    cat = SortingCategory(name="TestCat")
    db.add(cat)
    db.commit()
    
    response = client.delete("/api/v1/ai/sorting/categories/TestCat")
    assert response.status_code == 200


def test_delete_sorting_category_not_found(client):
    response = client.delete("/api/v1/ai/sorting/categories/GhostCategory")
    assert response.status_code == 404


def test_delete_sorting_rule_success(client, db):
    from app.models import SortingCategory, SortingRule
    cat = SortingCategory(name="Ceramic")
    db.add(cat)
    db.commit()
    rule = SortingRule(object_name="Vase", bin_name="Ceramic")
    db.add(rule)
    db.commit()
    
    response = client.delete(f"/api/v1/ai/sorting/rules/{rule.id}")
    assert response.status_code == 200


def test_delete_sorting_rule_not_found(client):
    response = client.delete("/api/v1/ai/sorting/rules/99999")
    assert response.status_code == 404


def test_simulate_sorting_success(client, db):
    from app.models import SortingCategory, SortingRule
    cat = SortingCategory(name="Metal")
    db.add(cat)
    db.commit()
    rule = SortingRule(object_name="Bolt", bin_name="Metal")
    db.add(rule)
    db.commit()
    
    from app.core.state import sorting_assistant
    sorting_assistant.run_sorting_step = MagicMock(return_value={
        "step": "pick",
        "object": "Bolt",
        "target_bin": "Metal"
    })
    
    response = client.post("/api/v1/ai/sorting/simulate")
    assert response.status_code == 200


def test_simulate_sorting_no_rules(client):
    # Patch db to return empty rules
    with patch("app.api.routes.sorting.get_sorting_rules_db", return_value=[]):
        response = client.post("/api/v1/ai/sorting/simulate")
    assert response.status_code == 400
    assert "No sorting rules" in response.json()["detail"]


def test_simulate_sorting_fails(client, db):
    from app.models import SortingCategory, SortingRule
    cat = SortingCategory(name="Paper")
    db.add(cat)
    db.commit()
    rule = SortingRule(object_name="Sheet", bin_name="Paper")
    db.add(rule)
    db.commit()
    
    from app.core.state import sorting_assistant
    sorting_assistant.run_sorting_step = MagicMock(return_value=None)
    
    response = client.post("/api/v1/ai/sorting/simulate")
    assert response.status_code == 400
