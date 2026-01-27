import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, date

from src.models import Therapy, Activity, PatientProfile, CaregiverProfile, Note

DATA_DIR = Path("data")
logger = logging.getLogger("kmchat.km")
class KnowledgeManager:
    def __init__(
        self,
        patient_id: str = None,
        caregiver_id: str = None,
        auto_discover: bool = False,
    ):
        self.therapy: Optional[Therapy] = None
        self.patient_profile: Optional[PatientProfile] = None
        self.caregiver_profile: Optional[CaregiverProfile] = None
        
        # Discovery automatico solo se richiesto
        if auto_discover:
            self.current_patient_id = patient_id or self._discover_first_id("patients")
            self.current_caregiver_id = caregiver_id or self._discover_first_id("caregivers")
        else:
            self.current_patient_id = patient_id
            self.current_caregiver_id = caregiver_id
        
        if self.current_patient_id and self.current_caregiver_id:
            self.load_data()
        elif auto_discover or patient_id or caregiver_id:
            logger.warning("Nessun contesto iniziale completo trovato. Usare set_context().")

    def _discover_first_id(self, folder_name: str) -> Optional[str]:
        target_dir = DATA_DIR / folder_name
        if target_dir.exists():
            files = sorted(target_dir.glob("*.json"))
            if files:
                return files[0].stem
        return None

    def _get_patient_file(self, pid: str) -> Path:
        return DATA_DIR / "patients" / f"{pid}.json"

    def _get_caregiver_file(self, cid: str) -> Path:
        return DATA_DIR / "caregivers" / f"{cid}.json"

    def _get_therapy_file(self, pid: str) -> Path:
        return DATA_DIR / "therapies" / f"{pid}.json"

    def set_context(self, patient_id: str, caregiver_id: str):
        self.current_patient_id = patient_id
        self.current_caregiver_id = caregiver_id
        self.load_data()

    def get_available_users(self):
        patients = []
        caregivers = []
        if (DATA_DIR / "patients").exists():
            for f in (DATA_DIR / "patients").glob("*.json"):
                try:
                    data = json.loads(f.read_text())
                    patients.append({"id": data.get("patient_id") or f.stem, "name": data.get("name")})
                except: pass
        if (DATA_DIR / "caregivers").exists():
            for f in (DATA_DIR / "caregivers").glob("*.json"):
                try:
                    data = json.loads(f.read_text())
                    caregivers.append({"id": data.get("caregiver_id") or f.stem, "name": data.get("name")})
                except: pass
        return {"patients": patients, "caregivers": caregivers}

    def find_patient_id_by_name(self, name: str) -> Optional[str]:
        if not name:
            return None
        target = name.strip().lower()
        if (DATA_DIR / "patients").exists():
            for f in (DATA_DIR / "patients").glob("*.json"):
                try:
                    data = json.loads(f.read_text())
                    if str(data.get("name", "")).strip().lower() == target:
                        return str(data.get("patient_id") or f.stem)
                except Exception:
                    continue
        return None

    def find_caregiver_id_by_name(self, name: str) -> Optional[str]:
        if not name:
            return None
        target = name.strip().lower()
        if (DATA_DIR / "caregivers").exists():
            for f in (DATA_DIR / "caregivers").glob("*.json"):
                try:
                    data = json.loads(f.read_text())
                    if str(data.get("name", "")).strip().lower() == target:
                        return str(data.get("caregiver_id") or f.stem)
                except Exception:
                    continue
        return None

    def load_data(self):
        if not self.current_patient_id:
            return

        p_file = self._get_patient_file(self.current_patient_id)
        # Default caregiver se non presente (può capitare in test parziali)
        c_id = self.current_caregiver_id or "unknown"
        c_file = self._get_caregiver_file(c_id)
        t_file = self._get_therapy_file(self.current_patient_id)

        # Therapy
        if t_file.exists():
            with open(t_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.therapy = Therapy(patient_id=self.current_patient_id, activities=[Activity(**a) for a in data])
                else:
                    self.therapy = Therapy(**data)
        else:
            self.therapy = Therapy(patient_id=self.current_patient_id, activities=[])

        # Patient Profile
        if p_file.exists():
            with open(p_file, "r") as f:
                self.patient_profile = PatientProfile(**json.load(f))
        else:
            # Fallback temporaneo se non esiste
            self.patient_profile = PatientProfile(patient_id=self.current_patient_id, name="Sconosciuto")

        # Caregiver Profile
        if c_file.exists():
            with open(c_file, "r") as f:
                self.caregiver_profile = CaregiverProfile(**json.load(f))
        else:
            self.caregiver_profile = CaregiverProfile(caregiver_id=c_id, name="Sconosciuto")

    def save_data(self):
        if self.therapy:
            t_file = self._get_therapy_file(self.current_patient_id)
            with open(t_file, "w") as f:
                f.write(self.therapy.model_dump_json(indent=4))

    def save_knowledge_note(self, category: str, content: str, day: str = None) -> str:
        target_profile = None
        save_path = None
        category = category.lower()
        
        if 'patient' in category or category in ['habits', 'preferences', 'conditions']:
            if not self.patient_profile: return "Errore: Profilo paziente non caricato."
            target_profile = self.patient_profile
            save_path = self._get_patient_file(self.current_patient_id)
        elif 'caregiver' in category:
            if not self.caregiver_profile: return "Errore: Profilo caregiver non caricato."
            target_profile = self.caregiver_profile
            save_path = self._get_caregiver_file(self.current_caregiver_id)
        else:
            return f"Categoria '{category}' non valida."

        # Gestione campi specifici per il paziente
        if target_profile == self.patient_profile:
            if category == 'habits':
                if content not in target_profile.habits:
                    target_profile.habits.append(content)
                    with open(save_path, "w") as f: f.write(target_profile.model_dump_json(indent=4))
                    return "Abitudine salvata."
                return "Abitudine già presente."
            elif category == 'preferences':
                if content not in target_profile.preferences:
                    target_profile.preferences.append(content)
                    with open(save_path, "w") as f: f.write(target_profile.model_dump_json(indent=4))
                    return "Preferenza salvata."
                return "Preferenza già presente."
            elif category == 'conditions':
                if content not in target_profile.medical_conditions:
                    target_profile.medical_conditions.append(content)
                    with open(save_path, "w") as f: f.write(target_profile.model_dump_json(indent=4))
                    return "Condizione medica salvata."
                return "Condizione medica già presente."

        # Deduplicazione Note
        for note in target_profile.notes:
            if note.content == content and note.day == day:
                return "Nota già presente (duplicato ignorato)."

        new_note = Note(content=content, day=day)
        target_profile.notes.append(new_note)
        
        with open(save_path, "w") as f:
            f.write(target_profile.model_dump_json(indent=4))
            
        return f"Nota salvata correttamente (Giorno: {day or 'Sempre'})."

    def _parse_time_to_minutes(self, time_str: str) -> int:
        try:
            h, m = map(int, time_str.split(':'))
            return h * 60 + m
        except ValueError: return -1

    def _get_time_interval(self, time_str: str, duration_minutes: int | None = None) -> tuple[int, int]:
        if '-' in time_str:
            parts = time_str.split('-')
            start = self._parse_time_to_minutes(parts[0].strip())
            end = self._parse_time_to_minutes(parts[1].strip())
            return start, end
        start = self._parse_time_to_minutes(time_str.strip())
        if duration_minutes is not None and duration_minutes > 0:
            return start, start + duration_minutes
        return start, start + 30

    def _parse_date(self, date_str: str) -> Optional[date]:
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str).date()
        except ValueError:
            return None

    def _is_activity_active_on_date(self, activity: Activity, target_date: date) -> bool:
        if not target_date:
            return True
        start = self._parse_date(activity.valid_from) if activity.valid_from else None
        end = self._parse_date(activity.valid_until) if activity.valid_until else None
        if start and target_date < start:
            return False
        if end and target_date > end:
            return False
        return True

    def get_activities_by_day(self, day: str, date_str: str = None) -> List[Activity]:
        if not self.therapy or not day:
            return []
        target = day.strip()
        target_date = self._parse_date(date_str) if date_str else datetime.today().date()
        results = []
        for act in self.therapy.activities:
            for act_day in act.day_of_week:
                if act_day.strip() == target and self._is_activity_active_on_date(act, target_date):
                    results.append(act)
                    break
        return results

    def get_week_schedule(self) -> Dict[str, List[Activity]]:
        week = {
            "Lunedì": [],
            "Martedì": [],
            "Mercoledì": [],
            "Giovedì": [],
            "Venerdì": [],
            "Sabato": [],
            "Domenica": [],
        }
        if not self.therapy:
            return week
        for act in self.therapy.activities:
            for act_day in act.day_of_week:
                act_day_clean = act_day.strip()
                if act_day_clean in week:
                    week[act_day_clean].append(act)
        return week

    def get_activity_by_name_day(self, name: str, day: str) -> Optional[Activity]:
        if not self.therapy or not name or not day:
            return None
        target = day.strip()
        for act in self.therapy.activities:
            if act.name != name:
                continue
            for act_day in act.day_of_week:
                if act_day.strip() == target:
                    return act
        return None

    def check_temporal_conflict(self, new_activity: Activity) -> List[str]:
        conflicts = []
        try:
            new_start, new_end = self._get_time_interval(
                new_activity.time,
                new_activity.duration_minutes,
            )
        except: return ["Formato orario non valido"]

        for existing in self.therapy.activities:
            common_days = set(new_activity.day_of_week) & set(existing.day_of_week)
            if common_days:
                try:
                    ex_start, ex_end = self._get_time_interval(
                        existing.time,
                        existing.duration_minutes,
                    )
                    if new_start < ex_end and new_end > ex_start:
                         conflicts.append(f"Conflitto temporale con '{existing.name}' ({existing.time}) nei giorni {common_days}")
                except: continue
        return conflicts

    def check_removal_conflict(self, activity_to_remove: Activity) -> List[str]:
        conflicts = []
        days_removed = set(activity_to_remove.day_of_week)
        for existing in self.therapy.activities:
            if existing.activity_id == activity_to_remove.activity_id: continue
            if activity_to_remove.name in existing.dependencies:
                common_days = set(existing.day_of_week) & days_removed
                if common_days:
                    conflicts.append(f"Rimozione '{activity_to_remove.name}' rompe la dipendenza per '{existing.name}' nei giorni {common_days}")
        return conflicts

    def check_missing_dependencies(self, new_activity: Activity) -> List[str]:
        issues = []
        new_days = set(new_activity.day_of_week)
        try: new_start, _ = self._get_time_interval(new_activity.time)
        except: new_start = -1

        for dep_name in new_activity.dependencies:
            found_dependency = None
            for existing in self.therapy.activities:
                if existing.name == dep_name:
                    if set(existing.day_of_week) & new_days:
                        found_dependency = existing
                        break
            if not found_dependency:
                issues.append(f"Dipendenza mancante: '{dep_name}' non trovata nei giorni {new_activity.day_of_week}")
            elif new_start != -1:
                try:
                    dep_start, _ = self._get_time_interval(found_dependency.time)
                    if new_start < dep_start:
                        issues.append(f"Errore sequenza: '{new_activity.name}' ({new_activity.time}) inizia prima della dipendenza '{dep_name}' ({found_dependency.time})")
                except: pass
        return issues

    def remove_activity(self, activity_name: str, day: str, force: bool = False) -> str:
        target_act = None
        target_idx = -1
        day_clean = day.strip() if isinstance(day, str) else day
        for i, act in enumerate(self.therapy.activities):
            if act.name == activity_name and day_clean in act.day_of_week:
                target_act = act
                target_idx = i
                break
        
        if not target_act: return f"Attività '{activity_name}' non trovata per {day_clean}."

        conflicts = self.check_removal_conflict(target_act)
        if conflicts:
            msg = f"ATTENZIONE: La rimozione crea conflitti di dipendenza: {'; '.join(conflicts)}."
            if not force: return f"{msg} Aggiungi 'force=True' per procedere comunque."
            logger.warning(f"Forzatura rimozione nonostante conflitti: {conflicts}")

        if len(target_act.day_of_week) > 1:
            target_act.day_of_week.remove(day_clean)
            self.save_data()
            return f"Attività '{activity_name}' rimossa dal giorno {day}."
        else:
            self.therapy.activities.pop(target_idx)
            self.save_data()
            return f"Attività '{activity_name}' eliminata definitivamente."

    def check_update_conflicts(self, old_name: str, day: str, new_data: dict) -> List[str]:
        target_act = None
        for act in self.therapy.activities:
            if act.name == old_name and day in act.day_of_week:
                target_act = act
                break
        if not target_act: return [f"Attività '{old_name}' non trovata per {day}."]

        updated_act = target_act.model_copy(update=new_data)
        warnings = []
        
        # Check temporale
        if "time" in new_data or "day_of_week" in new_data:
            t_conflicts = self.check_temporal_conflict(updated_act)
            # Filtriamo i conflitti con se stesso (che è ovvio ci siano prima dell'update)
            real_conflicts = [c for c in t_conflicts if old_name not in c and target_act.activity_id not in c] 
            # Nota: check_temporal_conflict usa il nome per il messaggio, ma qui stiamo simulando.
            # Miglioramento: check_temporal_conflict dovrebbe escludere l'ID dell'attività che si sta modificando.
            if real_conflicts: warnings.append(f"Conflitti temporali: {real_conflicts}")

        # Check dipendenze (se rinomino)
        if "name" in new_data and new_data["name"] != old_name:
            deps_conflicts = self.check_removal_conflict(target_act)
            if deps_conflicts: warnings.append(f"Rinominare rompe dipendenze: {deps_conflicts}")
            
        return warnings

    def update_activity(self, old_name: str, day: str, new_data: dict, force: bool = False) -> str:
        target_act = None
        for act in self.therapy.activities:
            if act.name == old_name and day in act.day_of_week:
                target_act = act
                break
        if not target_act: return f"Attività '{old_name}' non trovata per {day}."

        warnings = self.check_update_conflicts(old_name, day, new_data)

        if warnings:
            msg = "; ".join([str(w) for w in warnings])
            if not force: return f"Impossibile modificare: {msg}. Usa 'force=True' per forzare."
            logger.warning(f"Forzatura modifica nonostante: {msg}")

        for key, val in new_data.items(): setattr(target_act, key, val)
        self.save_data()
        return f"Attività '{old_name}' modificata in '{target_act.name}' con successo."

    def add_activity(self, activity: Activity, force: bool = False) -> str:
        # Prevent exact duplicates (same name, time, overlapping day, and validity window)
        for existing in self.therapy.activities:
            if existing.name != activity.name:
                continue
            if existing.time != activity.time:
                continue
            if not set(existing.day_of_week) & set(activity.day_of_week):
                continue
            if existing.valid_from != activity.valid_from or existing.valid_until != activity.valid_until:
                continue
            return "Attività già presente."

        warnings = []
        t_conflicts = self.check_temporal_conflict(activity)
        if t_conflicts: warnings.extend(t_conflicts)
        dep_issues = self.check_missing_dependencies(activity)
        if dep_issues: warnings.extend(dep_issues)

        if warnings:
            msg = "; ".join(warnings)
            if not force: return f"Impossibile aggiungere: {msg}. Usa 'force=True' per forzare l'inserimento."
            logger.warning(f"Forzatura aggiunta nonostante: {msg}")

        self.therapy.activities.append(activity)
        self.save_data()
        logger.info("Attività aggiunta: %s (%s)", activity.name, activity.time)
        return "Attività aggiunta con successo (Forzata)." if force else "Attività aggiunta con successo."
