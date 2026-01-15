import sys
import os

# Aggiungi la root al path per importare i moduli
sys.path.append(os.getcwd())

from src.knowledge_manager import KnowledgeManager
from src.main import check_semantic_conflict, km

def test_memory_isolation():
    print("\n--- INIZIO TEST MEMORIA E CONTEXT SWITCHING ---")

    # 1. Carichiamo Paziente 01
    print("\n[1] Caricamento Paziente 01...")
    km.set_context("paziente_01", "caregiver_01")
    print(f"Paziente Attivo: {km.patient_profile.name} ({km.patient_profile.patient_id})")
    
    # 2. Salviamo un'informazione critica per P01
    print("[2] Salvataggio nota critica per Paziente 01: 'Allergia mortale alle noci'")
    km.save_knowledge_note("conditions", "Allergia mortale alle noci")
    
    # Verifica immediata in memoria
    has_allergy_p1 = "Allergia mortale alle noci" in km.patient_profile.medical_conditions
    print(f"   -> Memoria P1 aggiornata? {has_allergy_p1}")

    # 3. CAMBIO CONTESTO -> Paziente 02
    print("\n[3] SWITCH CONTESTO -> Paziente 02 (Luigi Verdi)...")
    km.set_context("paziente_02", "caregiver_01")
    print(f"Paziente Attivo: {km.patient_profile.name} ({km.patient_profile.patient_id})")

    # 4. Verifica Isolamento: P02 NON deve avere l'allergia
    has_allergy_p2 = "Allergia mortale alle noci" in km.patient_profile.medical_conditions
    print(f"[4] Verifica contaminazione memoria: P02 ha l'allergia? {has_allergy_p2}")
    if has_allergy_p2:
        print("   ❌ ERRORE: I dati del Paziente 1 sono rimasti in memoria!")
    else:
        print("   ✅ SUCCESSO: La memoria del Paziente 2 è pulita.")

    # 5. Salviamo un'altra info per P02
    print("[5] Salvataggio abitudine per P02: 'Gioca a tennis il martedì'")
    km.save_knowledge_note("habits", "Gioca a tennis il martedì")

    # 6. RITORNO AL CONTESTO ORIGINALE -> Paziente 01
    print("\n[6] SWITCH BACK -> Paziente 01...")
    km.set_context("paziente_01", "caregiver_01")
    
    # 7. Verifica Persistenza: P01 deve ancora avere l'allergia (ricaricata dal file)
    # Ricarichiamo esplicitamente per essere sicuri che legga dal disco se necessario
    km.load_data() 
    has_allergy_p1_reload = "Allergia mortale alle noci" in km.patient_profile.medical_conditions
    print(f"[7] Verifica persistenza P01: L'allergia è ancora lì? {has_allergy_p1_reload}")
    
    if has_allergy_p1_reload:
        print("   ✅ SUCCESSO: Il sistema ha ricordato i dati del Paziente 1 dopo il cambio contesto.")
    else:
        print("   ❌ ERRORE: I dati del Paziente 1 sono andati persi!")

    # 8. Check Semantico Incrociato (Simulato)
    # Proviamo a dare noci a P1
    print("\n[8] Test Semantico su P1: 'Mangiare torta alle noci'")
    # Simuliamo il check chiamando la logica interna (senza LLM per velocità, o con se configurato)
    # Qui stampiamo solo le regole che verrebbero passate all'LLM
    rules = km.patient_profile.medical_conditions
    print(f"   Regole attive per il check: {rules}")
    if "Allergia mortale alle noci" in rules:
        print("   ✅ SUCCESSO: L'allergia è presente nelle regole di validazione.")
    else:
         print("   ❌ ERRORE: Regola mancante.")

if __name__ == "__main__":
    test_memory_isolation()
