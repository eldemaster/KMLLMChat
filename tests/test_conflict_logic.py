import pytest
from src.knowledge_manager import KnowledgeManager
from src.models import Activity, Therapy

@pytest.fixture
def km_empty():
    """KnowledgeManager con terapia vuota per test isolati"""
    km = KnowledgeManager()
    km.therapy = Therapy(patient_id="test", activities=[])
    return km

def test_conflict_exact_match(km_empty):
    existing = Activity(
        activity_id="1", name="A", description="desc", 
        day_of_week=["Lunedì"], time="10:00-11:00", dependencies=[]
    )
    km_empty.therapy.activities.append(existing)
    
    new_act = Activity(
        activity_id="2", name="B", description="desc", 
        day_of_week=["Lunedì"], time="10:00-11:00", dependencies=[]
    )
    
    conflicts = km_empty.check_temporal_conflict(new_act)
    assert len(conflicts) > 0

def test_conflict_partial_overlap(km_empty):
    existing = Activity(
        activity_id="1", name="A", description="desc", 
        day_of_week=["Lunedì"], time="10:00-11:00", dependencies=[]
    )
    km_empty.therapy.activities.append(existing)
    
    # Sovrapposizione 10:30 - 11:30
    new_act = Activity(
        activity_id="2", name="B", description="desc", 
        day_of_week=["Lunedì"], time="10:30-11:30", dependencies=[]
    )
    
    conflicts = km_empty.check_temporal_conflict(new_act)
    assert len(conflicts) > 0, "Dovrebbe rilevare sovrapposizione parziale"

def test_no_conflict_adjacent(km_empty):
    existing = Activity(
        activity_id="1", name="A", description="desc", 
        day_of_week=["Lunedì"], time="10:00-11:00", dependencies=[]
    )
    km_empty.therapy.activities.append(existing)
    
    # Inizia quando l'altra finisce
    new_act = Activity(
        activity_id="2", name="B", description="desc", 
        day_of_week=["Lunedì"], time="11:00-12:00", dependencies=[]
    )
    
    conflicts = km_empty.check_temporal_conflict(new_act)
    assert len(conflicts) == 0, "Non dovrebbe esserci conflitto per orari adiacenti"

def test_conflict_single_time_format(km_empty):
    # Se il sistema supporta "HH:MM" come "HH:MM-HH:MM+delta" o puntuale
    # Per ora assumiamo che se passo "10:00" e esiste "10:00-11:00", c'è conflitto
    existing = Activity(
        activity_id="1", name="A", description="desc", 
        day_of_week=["Lunedì"], time="10:00-11:00", dependencies=[]
    )
    km_empty.therapy.activities.append(existing)
    
    new_act = Activity(
        activity_id="2", name="B", description="desc", 
        day_of_week=["Lunedì"], time="10:15", dependencies=[]
    )
    
    conflicts = km_empty.check_temporal_conflict(new_act)
    assert len(conflicts) > 0, "Un'attività puntuale dentro un intervallo dovrebbe confliggere"

def test_conflict_duration_minutes(km_empty):
    existing = Activity(
        activity_id="1", name="A", description="desc",
        day_of_week=["Lunedì"], time="10:00", duration_minutes=60, dependencies=[]
    )
    km_empty.therapy.activities.append(existing)

    new_act = Activity(
        activity_id="2", name="B", description="desc",
        day_of_week=["Lunedì"], time="10:30-11:00", dependencies=[]
    )

    conflicts = km_empty.check_temporal_conflict(new_act)
    assert len(conflicts) > 0, "La durata deve estendere l'intervallo per rilevare sovrapposizioni"
