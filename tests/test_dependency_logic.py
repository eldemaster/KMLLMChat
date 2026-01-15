import pytest
from src.knowledge_manager import KnowledgeManager
from src.models import Activity, Therapy

@pytest.fixture
def km_with_deps():
    """KnowledgeManager con attività collegate da dipendenze"""
    km = KnowledgeManager()
    
    act1 = Activity(
        activity_id="1", name="Colazione", description="Mangiare", 
        day_of_week=["Lunedì"], time="08:00", dependencies=[]
    )
    act2 = Activity(
        activity_id="2", name="Farmaco A", description="Prendere dopo colazione", 
        day_of_week=["Lunedì"], time="08:30", dependencies=["Colazione"]
    )
    
    km.therapy = Therapy(patient_id="test", activities=[act1, act2])
    return km

def test_remove_dependency_conflict(km_with_deps):
    # Provo a rimuovere "Colazione". "Farmaco A" dipende da essa.
    # Mi aspetto che il sistema rilevi che "Farmaco A" rimarrebbe orfano.
    
    # Cerco l'attività da rimuovere
    act_to_remove = next(a for a in km_with_deps.therapy.activities if a.name == "Colazione")
    
    conflicts = km_with_deps.check_removal_conflict(act_to_remove)
    assert len(conflicts) > 0
    assert "Farmaco A" in conflicts[0]

def test_remove_safe(km_with_deps):
    # Rimuovo "Farmaco A". Nessuno dipende da lui.
    act_to_remove = next(a for a in km_with_deps.therapy.activities if a.name == "Farmaco A")
    
    conflicts = km_with_deps.check_removal_conflict(act_to_remove)
    assert len(conflicts) == 0

def test_add_missing_dependency(km_with_deps):
    # Aggiungo attività che dipende da "Pranzo" (che non c'è)
    new_act = Activity(
        activity_id="3", name="Caffè", description="Dopo pranzo", 
        day_of_week=["Lunedì"], time="13:00", dependencies=["Pranzo"]
    )
    
def test_dependency_sequence_error(km_with_deps):
    # 'Colazione' è alle 08:00 (durata default 30m -> 08:30)
    
    # Caso 1: Attività PRIMA della dipendenza (07:00 < 08:00) -> Errore
    too_early = Activity(
        activity_id="4", name="Presto", description="...", 
        day_of_week=["Lunedì"], time="07:00", dependencies=["Colazione"]
    )
    issues = km_with_deps.check_missing_dependencies(too_early)
    assert len(issues) > 0
    assert "inizia prima della dipendenza" in issues[0]

    # Caso 2: Attività DOPO la dipendenza (09:00 > 08:00) -> OK
    ok_time = Activity(
        activity_id="5", name="Dopo", description="...", 
        day_of_week=["Lunedì"], time="09:00", dependencies=["Colazione"]
    )
    issues_ok = km_with_deps.check_missing_dependencies(ok_time)
    assert len(issues_ok) == 0
