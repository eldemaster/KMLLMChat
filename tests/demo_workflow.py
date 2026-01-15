import sys
import os
import time

# Add root to path
sys.path.append(os.getcwd())

from src.knowledge_manager import KnowledgeManager
from src.models import Activity, PatientProfile, CaregiverProfile
# We need to import the semantic check function. 
# Note: This requires Ollama to be running.
try:
    from src.main import check_semantic_conflict
except ImportError:
    print("Warning: Could not import check_semantic_conflict. Skipping LLM tests.")
    check_semantic_conflict = lambda x, y: None

def print_step(title):
    print(f"\n{'='*50}\nSTEP: {title}\n{'='*50}")

def run_demo():
    print("🚀 STARTING FULL SYSTEM DEMO 🚀")
    
    # 1. Setup Context
    print_step("1. Setup Context (Patient: Giuseppe, Caregiver: Dr. House)")
    km = KnowledgeManager()
    
    # Create fake profiles
    p_id = "demo_giuseppe"
    c_id = "demo_house"
    
    # Reset files for demo
    import json
    p_file = km._get_patient_file(p_id)
    c_file = km._get_caregiver_file(c_id)
    t_file = km._get_therapy_file(p_id)
    
    if p_file.exists(): p_file.unlink()
    if c_file.exists(): c_file.unlink()
    if t_file.exists(): t_file.unlink()
    
    # Set context which will create/load default empty ones, then we overwrite
    km.set_context(p_id, c_id)
    
    # Update Profile
    km.patient_profile.name = "Giuseppe Rossi"
    km.patient_profile.medical_conditions = ["Severe Heart Condition", "Arrhythmia"]
    km.patient_profile.notes = []
    km.save_knowledge_note("patient", "Must avoid high heart rate activities")
    print(f"Patient Profile Created: {km.patient_profile.name}, Conditions: {km.patient_profile.medical_conditions}")

    # 2. Add Base Therapy
    print_step("2. Define Base Therapy")
    act1 = Activity(
        activity_id="act_01",
        name="Take Heart Meds",
        description="Beta-blockers",
        day_of_week=["Monday", "Wednesday"],
        time="08:00",
        dependencies=[]
    )
    res = km.add_activity(act1)
    print(f"Adding 'Heart Meds': {res}")

    act2 = Activity(
        activity_id="act_02",
        name="Light Walk",
        description="Slow walking in the park",
        day_of_week=["Monday"],
        time="09:00-09:30",
        dependencies=["Take Heart Meds"]
    )
    res = km.add_activity(act2)
    print(f"Adding 'Light Walk': {res}")

    # 3. Test Temporal Conflict
    print_step("3. Test Temporal Conflict")
    act_conflict = Activity(
        activity_id="act_conflict",
        name="Nap",
        description="Resting",
        day_of_week=["Monday"],
        time="09:15-10:00", # Overlaps with Light Walk (09:00-09:30)
        dependencies=[]
    )
    conflicts = km.check_temporal_conflict(act_conflict)
    if conflicts:
        print(f"✅ Conflict Detected correctly: {conflicts[0]}")
    else:
        print("❌ FAIL: No conflict detected!")

    # 4. Test Dependency Constraint
    print_step("4. Test Dependency Constraint")
    # Try to remove "Take Heart Meds" which "Light Walk" depends on
    msg = km.remove_activity("Take Heart Meds", "Monday")
    print(f"Attempting removal: {msg}")
    if "ATTENZIONE" in msg and "dipendenza" in msg.lower():
        print("✅ Dependency Warning Triggered correctly.")
    else:
        print("❌ FAIL: No dependency warning.")

    # 5. Test Semantic Conflict (LLM)
    print_step("5. Test Semantic Conflict (LLM Check)")
    print("Action: Adding 'Crossfit Session' (High intensity)")
    print("Patient Condition: 'Severe Heart Condition', 'Must avoid high heart rate'")
    
    proposed_name = "Crossfit Session"
    proposed_desc = "High intensity interval training with weights"
    
    # We call the function from main.py which uses the LLM
    # Note: We need to ensure km in main.py points to our km instance or verify main.py uses its own.
    # Actually, main.py instantiates its own 'km'. To test correctly with main's logic, 
    # we should ideally inject our context into main's km, or mock it.
    # Let's mock the profile in the main module if possible, or just rely on main.py loading the files we just wrote!
    # Since we saved the data to disk (km.save_knowledge_note writes to disk), main.py's KM will load it.
    
    # We need to re-init main's KM or force context switch
    import src.main
    from llama_index.llms.ollama import Ollama
    from llama_index.core import Settings
    
    # Initialize LLM for the test environment
    Settings.llm = Ollama(model="kmchat-14b", request_timeout=120.0)
    
    src.main.km.set_context(p_id, c_id) # Reloads from disk
    
    warning = src.main.check_semantic_conflict(proposed_name, proposed_desc)
    print(f"LLM Response: {warning}")
    
    if warning and "YES" in warning:
        print("✅ Semantic Conflict Detected correctly by LLM.")
    else:
        print("❌ FAIL: LLM did not detect conflict (or model is too weak/distracted).")

    print_step("DEMO COMPLETE")

if __name__ == "__main__":
    run_demo()
