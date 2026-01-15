import sys
import os
from pathlib import Path
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.knowledge_manager import KnowledgeManager
from src.models import Activity

def test_persistence():
    print("--- TEST PERSISTENZA FILES JSON ---")
    
    # 1. Setup
    km = KnowledgeManager()
    # Force context to a demo patient to avoid messing up real data if possible, 
    # but let's use the current one to be sure it works where the user is working.
    # We'll use a specific ID to be safe.
    test_pid = "test_persistence_patient"
    test_cid = "test_persistence_caregiver"
    
    # Create dummy files for context loading
    (Path("data/patients") / f"{test_pid}.json").write_text(json.dumps({"patient_id": test_pid, "name": "Test Patient", "medical_conditions": [], "preferences": [], "habits": [], "notes": []}))
    (Path("data/caregivers") / f"{test_cid}.json").write_text(json.dumps({"caregiver_id": test_cid, "name": "Test Caregiver", "notes": []}))
    
    km.set_context(test_pid, test_cid)
    
    # 2. Add Activity
    act_name = "TEST_ACTIVITY_PERSISTENCE"
    new_activity = Activity(
        activity_id="test_act_1",
        name=act_name,
        description="Testing file save",
        day_of_week=["Lunedì"],
        time="10:00",
        dependencies=[]
    )
    
    print(f"Adding activity '{act_name}' in memory...")
    km.add_activity(new_activity, force=True)
    
    # 3. Verify File
    therapy_file = Path("data/therapies") / f"{test_pid}.json"
    if not therapy_file.exists():
        print(f"❌ FAIL: File {therapy_file} non creato!")
        return
        
    content = therapy_file.read_text()
    if act_name in content:
        print(f"✅ SUCCESS: L'attività '{act_name}' è stata scritta correttamente su disco in {therapy_file}.")
    else:
        print(f"❌ FAIL: Il file esiste ma non contiene l'attività! Contenuto:\n{content}")

    # Cleanup
    try:
        therapy_file.unlink()
        (Path("data/patients") / f"{test_pid}.json").unlink()
        (Path("data/caregivers") / f"{test_cid}.json").unlink()
        print("Cleanup completato.")
    except Exception as e:
        print(f"Cleanup error: {e}")

if __name__ == "__main__":
    test_persistence()
