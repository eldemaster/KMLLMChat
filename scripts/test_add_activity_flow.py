import sys
import os
import json
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.knowledge_manager import KnowledgeManager
from src.models import Activity

def test_full_add_flow():
    print("--- TEST AGGIUNTA E VERIFICA ATTIVITÀ ---")
    
    # 1. Setup ambiente pulito
    pid = "test_add_patient"
    cid = "test_add_caregiver"
    
    # File JSON di partenza vuoti/base
    (Path("data/patients") / f"{pid}.json").write_text(json.dumps({"patient_id": pid, "name": "Test Patient", "medical_conditions": [], "preferences": [], "habits": [], "notes": []}))
    (Path("data/caregivers") / f"{cid}.json").write_text(json.dumps({"caregiver_id": cid, "name": "Test Caregiver", "notes": []}))
    # Terapia vuota
    (Path("data/therapies") / f"{pid}.json").write_text(json.dumps([])) # Lista vuota o oggetto vuoto

    km = KnowledgeManager()
    km.set_context(pid, cid)
    
    print(f"1. Stato iniziale attività Lunedì: {len(km.get_activities_by_day('Lunedì'))}")

    # 2. Aggiunta Attività (simuliamo la chiamata del tool)
    new_act = Activity(
        activity_id="act_fisio_1",
        name="Fisioterapia",
        description="Esercizi gambe",
        day_of_week=["Lunedì"],
        time="15:00",
        dependencies=[]
    )
    
    print("2. Chiamata add_activity(..., force=True)...")
    res = km.add_activity(new_act, force=True)
    print(f"   Risultato KM: {res}")

    # 3. Verifica Immediata in Memoria
    activities = km.get_activities_by_day("Lunedì")
    found = any(a.name == "Fisioterapia" for a in activities)
    print(f"3. Verifica in Memoria (get_activities_by_day): {'✅ TROVATA' if found else '❌ NON TROVATA'}")
    
    if not found:
        print("   DEBUG: Attività presenti in memoria:")
        for a in activities: print(f"   - {a.name}")

    # 4. Verifica su Disco (JSON)
    print("4. Verifica su Disco (JSON)...")
    file_content = (Path("data/therapies") / f"{pid}.json").read_text()
    if "Fisioterapia" in file_content and "15:00" in file_content:
        print("   ✅ FILE AGGIORNATO CORRETTAMENTE")
    else:
        print("   ❌ FILE NON AGGIORNATO")
        print(f"   Contenuto File: {file_content}")

    # 5. Verifica Ricaricamento (Simuliamo nuovo KM)
    print("5. Verifica Ricaricamento (Nuova istanza KM)...")
    km2 = KnowledgeManager()
    km2.set_context(pid, cid)
    activities2 = km2.get_activities_by_day("Lunedì")
    found2 = any(a.name == "Fisioterapia" for a in activities2)
    print(f"   Risultato: {'✅ PERSISTENZA OK' if found2 else '❌ PERSISTENZA KO'}")

    # Cleanup
    try:
        (Path("data/patients") / f"{pid}.json").unlink()
        (Path("data/caregivers") / f"{cid}.json").unlink()
        (Path("data/therapies") / f"{pid}.json").unlink()
    except: pass

if __name__ == "__main__":
    test_full_add_flow()
