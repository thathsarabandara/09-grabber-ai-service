import datetime
from typing import Dict, List

class PickPlaceSimulator:
    def __init__(self):
        pass

    def run_simulation(
        self,
        strategy: str,
        selection_rule: str,
        pick_min_x: float,
        pick_max_x: float,
        drop_min_x: float,
        drop_max_x: float,
        cube_force: str
    ) -> Dict:
        """Generate frame-by-frame 2D kinematic arm coordinate path and log messages."""
        path = []
        for frame in range(121):
            armX, armY = 100, 100
            objX, objY = 100, 190
            
            if frame < 30:
                armY = 100 + (frame / 30.0) * 80.0
            elif frame < 50:
                armY = 180.0
            elif frame < 80:
                diff = (frame - 50.0) / 30.0
                armY = 180.0 - diff * 100.0
                objY = 190.0 - diff * 100.0
                armX = 100.0 + diff * 240.0
                objX = 100.0 + diff * 240.0
            elif frame < 100:
                diff = (frame - 80.0) / 20.0
                armY = 80.0 + diff * 100.0
                objY = 90.0 + diff * 100.0
                armX = 340.0
                objX = 340.0
            else:
                diff = (frame - 100.0) / 20.0
                armY = 180.0 - diff * 80.0
                armX = 340.0 - diff * 240.0
                objX = 340.0
                objY = 190.0
                
            path.append({
                "frame": frame,
                "armX": armX,
                "armY": armY,
                "objX": objX,
                "objY": objY
            })
            
        log_entries = [
            f"CMD: PICK target_id=Cube_01 (Grasp Force: {cube_force})",
            f"IK_SOLVER: theta=[45, 90, -12, 0] (Approach: {strategy})",
            f"GRIP: pressure_sensor={cube_force}",
            f"CMD: DROP target_id=Bin_Green (Rule: {selection_rule})",
            f"IK_SOLVER: theta=[0, 30, 10, 0]",
            f"Pick success. Cycle time 4.2s (Bounds: [{pick_min_x}, {pick_max_x}] to [{drop_min_x}, {drop_max_x}])"
        ]
        
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        timestamped_logs = [f"[{now_str}] {log}" for log in log_entries]
        
        return {
            "path_coords": path,
            "log_entries": timestamped_logs
        }
