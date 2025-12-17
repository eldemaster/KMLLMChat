import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from src.models import Therapy, Activity, PatientProfile, CaregiverProfile
# Importiamo le costanti o configurazioni se necessario

DATA_FILE = Path("data/therapy.json")
PATIENT_FILE = Path("data/patient_profile.json")
CAREGIVER_FILE = Path("data/caregiver_profile.json")

logger = logging.getLogger("kmchat.km")

class KnowledgeManager:
    def __init__(self):
        self.therapy: Optional[Therapy] = None
        self.patient_profile: Optional[PatientProfile] = None
        self.caregiver_profile: Optional[CaregiverProfile] = None
        self.load_data()

    def load_data(self):
        """Carica terapia e profili dai file JSON"""
        # 1. Carica Terapia
        if DATA_FILE.exists():
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.therapy = Therapy(patient_id="paziente_01", activities=[Activity(**a) for a in data])
                else:
                    self.therapy = Therapy(**data)
            logger.info("Terapia caricata.")
        else:
            # Dati di default
            self.therapy = Therapy(
                patient_id="paziente_01", 
                activities=[
                    Activity(activity_id="act_001", name="Assunzione Aulin", description="Assumi l'Aulin con acqua", day_of_week=["Lunedì", "Mercoledì", "Venerdì"], time="08:00", dependencies=["Colazione"]),
                    Activity(activity_id="act_002", name="Riabilitazione ginocchio", description="Fisioterapia ginocchio", day_of_week=["Martedì", "Giovedì", "Venerdì"], time="09:00-09:30", dependencies=[]),
                    Activity(activity_id="act_003", name="Esercizi cognitivi", description="Leggere o esercizi psicologa", day_of_week=["Lunedì", "Martedì", "Giovedì", "Venerdì"], time="10:00-10:30", dependencies=[])
                ]
            )
            self.save_data()
            logger.info("Terapia di default creata.")

        # 2. Carica Profilo Paziente
        if PATIENT_FILE.exists():
            with open(PATIENT_FILE, "r") as f:
                self.patient_profile = PatientProfile(**json.load(f))
            logger.info(f"Profilo Paziente caricato: {self.patient_profile.name}")

        # 3. Carica Profilo Caregiver
        if CAREGIVER_FILE.exists():
            with open(CAREGIVER_FILE, "r") as f:
                self.caregiver_profile = CaregiverProfile(**json.load(f))
            logger.info(f"Profilo Caregiver caricato: {self.caregiver_profile.name}")

    def save_data(self):
        """Salva lo stato corrente su JSON"""
        if self.therapy:
            with open(DATA_FILE, "w") as f:
                # model_dump_json è il metodo Pydantic v2, usare .json() per v1
                f.write(self.therapy.model_dump_json(indent=4))
            logger.info("Terapia salvata su %s (%d attività)", DATA_FILE, len(self.therapy.activities))

    def get_activities_by_day(self, day: str) -> List[Activity]:
        """Restituisce le attività per un dato giorno"""
        if not self.therapy:
            return []
        matches = [act for act in self.therapy.activities if day in act.day_of_week]
        logger.info("Richieste attività per %s: trovate %d", day, len(matches))
        return matches

    def _parse_time_to_minutes(self, time_str: str) -> int:
        """Converte 'HH:MM' in minuti dall'inizio della giornata."""
        try:
            h, m = map(int, time_str.split(':'))
            return h * 60 + m
        except ValueError:
            return -1 # Formato non valido

    def _get_time_interval(self, time_str: str) -> tuple[int, int]:
        """
        Restituisce (start_min, end_min) da una stringa.
        Supporta 'HH:MM' (durata default 30 min) e 'HH:MM-HH:MM'.
        """
        if '-' in time_str:
            parts = time_str.split('-')
            start = self._parse_time_to_minutes(parts[0].strip())
            end = self._parse_time_to_minutes(parts[1].strip())
            return start, end
        else:
            start = self._parse_time_to_minutes(time_str.strip())
            # Default duration: 30 minutes
            return start, start + 30

    def check_temporal_conflict(self, new_activity: Activity) -> List[str]:
        """
        Controlla se ci sono sovrapposizioni temporali.
        Restituisce una lista di messaggi di conflitto.
        """
        conflicts = []
        
        # Parsiamo il tempo della nuova attività
        try:
            new_start, new_end = self._get_time_interval(new_activity.time)
        except Exception:
            return ["Formato orario non valido (usa HH:MM o HH:MM-HH:MM)"]

        for existing in self.therapy.activities:
            # Controlla se i giorni si sovrappongono
            common_days = set(new_activity.day_of_week) & set(existing.day_of_week)
            if common_days:
                try:
                    ex_start, ex_end = self._get_time_interval(existing.time)
                    
                    # Logica di overlap: (StartA < EndB) and (EndA > StartB)
                    if new_start < ex_end and new_end > ex_start:
                         conflicts.append(f"Conflitto temporale con '{existing.name}' ({existing.time}) nei giorni {common_days}")
                except Exception:
                    # Ignoriamo errori di parsing sulle attività esistenti per robustezza
                    continue
        
        return conflicts

    def check_removal_conflict(self, activity_to_remove: Activity) -> List[str]:
        """
        Controlla se la rimozione di un'attività crea conflitti di dipendenza.
        Esempio: Rimuovo 'Colazione', ma 'Farmaco A' richiede 'Colazione'.
        """
        conflicts = []
        days_removed = set(activity_to_remove.day_of_week)
        
        for existing in self.therapy.activities:
            if existing.activity_id == activity_to_remove.activity_id:
                continue
                
            # Se l'attività esistente dipende da quella che sto rimuovendo
            if activity_to_remove.name in existing.dependencies:
                # Verifico se i giorni coincidono
                common_days = set(existing.day_of_week) & days_removed
                if common_days:
                    conflicts.append(f"Rimozione '{activity_to_remove.name}' rompe la dipendenza per '{existing.name}' nei giorni {common_days}")
        
        return conflicts

    def check_missing_dependencies(self, new_activity: Activity) -> List[str]:
        """
        Controlla se le dipendenze dichiarate per la nuova attività sono soddisfatte.
        """
        issues = []
        new_days = set(new_activity.day_of_week)
        
        for dep_name in new_activity.dependencies:
            # Cerco se esiste un'attività con questo nome nei giorni richiesti
            is_satisfied = False
            for existing in self.therapy.activities:
                if existing.name == dep_name:
                    if set(existing.day_of_week) & new_days:
                        is_satisfied = True
                        break
            
            if not is_satisfied:
                issues.append(f"Dipendenza mancante: '{dep_name}' non trovata nei giorni {new_activity.day_of_week}")
        
        return issues

    def save_knowledge_note(self, category: str, content: str) -> str:
        """
        Salva una nota di conoscenza estratta nel profilo appropriato.
        category: 'patient' o 'caregiver'
        """
        if category.lower() == 'patient':
            if not self.patient_profile:
                return "Errore: Profilo paziente non caricato."
            self.patient_profile.notes.append(content)
            # Salva su file
            with open(PATIENT_FILE, "w") as f:
                f.write(self.patient_profile.model_dump_json(indent=4))
            logger.info(f"Nota salvata per paziente: {content}")
            return f"Nota salvata nel profilo paziente: {content}"
            
        elif category.lower() == 'caregiver':
            if not self.caregiver_profile:
                return "Errore: Profilo caregiver non caricato."
            self.caregiver_profile.notes.append(content)
            with open(CAREGIVER_FILE, "w") as f:
                f.write(self.caregiver_profile.model_dump_json(indent=4))
            logger.info(f"Nota salvata per caregiver: {content}")
            return f"Nota salvata nel profilo caregiver: {content}"
            
        else:
            return f"Categoria '{category}' non valida. Usa 'patient' o 'caregiver'."

    def add_activity(self, activity: Activity) -> str:
        """Aggiunge un'attività se non ci sono conflitti bloccanti"""
        # 1. Conflitti Temporali
        conflicts = self.check_temporal_conflict(activity)
        if conflicts:
            logger.warning("Conflitto temporale nell'aggiunta di %s: %s", activity.name, conflicts)
            return f"Impossibile aggiungere: {'; '.join(conflicts)}"
        
        # 2. Conflitti di Dipendenza (Mancanza precondizione)
        dep_issues = self.check_missing_dependencies(activity)
        if dep_issues:
            logger.warning("Dipendenze mancanti per %s: %s", activity.name, dep_issues)
            return f"Attenzione, dipendenze non soddisfatte: {'; '.join(dep_issues)}"

        self.therapy.activities.append(activity)
        self.save_data()
        logger.info("Attività aggiunta: %s (%s)", activity.name, activity.time)
        return "Attività aggiunta con successo."

# Istanza globale per test rapidi
km = KnowledgeManager()
