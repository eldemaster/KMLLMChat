from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class Note(BaseModel):
    """
    Rappresenta una nota di conoscenza estratta.
    """
    content: str = Field(..., description="Il contenuto della nota")
    day: Optional[str] = Field(None, description="Giorno di validità (es. 'Lunedì'). Se None, vale sempre.")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class Activity(BaseModel):
    """
    Rappresenta una singola attività all'interno della terapia.
    """
    activity_id: str = Field(..., description="Identificativo univoco dell'attività (es. act_001)")
    name: str = Field(..., description="Nome breve dell'attività")
    description: str = Field(..., description="Descrizione dettagliata dell'attività")
    day_of_week: List[str] = Field(..., description="Giorni della settimana in cui svolgere l'attività")
    time: str = Field(..., description="Orario specifico (es. 08:00) o finestra temporale (es. 09:00-09:30)")
    duration_minutes: Optional[int] = Field(None, description="Durata in minuti se l'orario è un singolo orario")
    dependencies: List[str] = Field(default_factory=list, description="Lista di attività o eventi da cui questa attività dipende")
    valid_from: Optional[str] = Field(None, description="Data inizio validità (YYYY-MM-DD)")
    valid_until: Optional[str] = Field(None, description="Data fine validità (YYYY-MM-DD)")

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
    habits: List[str] = Field(default_factory=list, description="Es. Beve caffè dopo pranzo")
    notes: List[Note] = Field(default_factory=list, description="Note strutturate con giorno opzionale")

class CaregiverProfile(BaseModel):
    """
    Informazioni relative al caregiver per adattare lo stile di conversazione.
    """
    caregiver_id: str
    name: str
    role: Optional[str] = None
    semantic_preferences: List[str] = Field(default_factory=list, description="Es. 'Aulin' inteso come granulare")
    notes: List[Note] = Field(default_factory=list, description="Note strutturate con giorno opzionale")
