from typing import List, Optional
from pydantic import BaseModel, Field

class Activity(BaseModel):
    """
    Rappresenta una singola attività all'interno della terapia.
    Basato sull'esempio del PDF:
    {
        "activity_id": "act_001",
        "name": "Assunzione Aulin",
        "description": "Assumi l'Aulin con acqua",
        "day_of_week": ["Lunedì", "Mercoledì", "Venerdì"],
        "time": "08:00",
        "dependencies": ["Colazione"]
    }
    """
    activity_id: str = Field(..., description="Identificativo univoco dell'attività (es. act_001)")
    name: str = Field(..., description="Nome breve dell'attività")
    description: str = Field(..., description="Descrizione dettagliata dell'attività")
    day_of_week: List[str] = Field(..., description="Giorni della settimana in cui svolgere l'attività")
    time: str = Field(..., description="Orario specifico (es. 08:00) o finestra temporale (es. 09:00-09:30)")
    dependencies: List[str] = Field(default_factory=list, description="Lista di attività o eventi da cui questa attività dipende")

class Therapy(BaseModel):
    """
    Rappresenta l'insieme delle attività che compongono la terapia di un paziente.
    """
    patient_id: str = Field(..., description="ID del paziente a cui è associata la terapia")
    activities: List[Activity] = Field(default_factory=list, description="Lista delle attività previste")

class PatientProfile(BaseModel):
    """
    Informazioni relative al paziente (salute, abitudini, preferenze).
    """
    patient_id: str
    name: str
    medical_conditions: List[str] = Field(default_factory=list, description="Es. Diabete, Celiachia")
    preferences: List[str] = Field(default_factory=list, description="Es. Riposino alle 15:00")
    notes: List[str] = Field(default_factory=list, description="Note libere e conoscenza estratta dalla chat")

class CaregiverProfile(BaseModel):
    """
    Informazioni relative al caregiver per adattare lo stile di conversazione.
    """
    caregiver_id: str
    name: str
    role: Optional[str] = None
    semantic_preferences: List[str] = Field(default_factory=list, description="Es. 'Aulin' inteso come granulare")
    notes: List[str] = Field(default_factory=list, description="Note libere e conoscenza estratta dalla chat")
