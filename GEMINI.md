# GEMINI Project Context: KMChat

## Project Overview

**Project Title:** [25-KMChat] Managing the Knowledge Exchanged in a Chat with LLMs

This project aims to develop an LLM-based system for the healthcare sector to assist caregivers (e.g., nurses) in managing patient therapies. The core function of the system is to interpret caregiver requests, manage therapy modifications, and ensure consistency by detecting conflicts between existing knowledge (pre-defined weekly therapies) and new knowledge (daily adjustments).

## Key Files

*   **`Progetto Digital Transformation - Alessandro De Martini.pdf`**: The official project specification document. It details the scenario, system capabilities, knowledge base requirements, and examples of conflicts.

## System Requirements

The system must act as a support tool for caregivers, providing the following capabilities:

1.  **Knowledge Retrieval & Comparison:** Retrieve relevant information from the knowledge base using **RAG (Retrieval-Augmented Generation)** and compare it with new information from the interaction.
2.  **Conflict Detection:** Identify inconsistencies, conflicts, or missing information between the new input and the existing system state (e.g., concurrent activities, missing preconditions).
3.  **Conflict Resolution Support:** Present detected conflicts to the caregiver for resolution. The system **must not** resolve conflicts automatically; the human remains the final decision-maker.
4.  **Knowledge Extraction:** Analyze conversations to extract and structure relevant data regarding:
    *   **Therapies:** Activities, schedules, dependencies.
    *   **Patients:** Health data, preferences, habits.
    *   **Caregivers:** Personal data, linguistic preferences.
5.  **Controlled Knowledge Update:** Update the knowledge base (e.g., vector database like Chroma) with validated information.

## Data Structure Examples

The project specification provides JSON examples for therapy activities, including fields like:
*   `activity_id`
*   `name`
*   `description`
*   `day_of_week`
*   `time`
*   `dependencies`

## Development Context

This directory currently contains the project requirements. Future work will involve implementing the described system, likely involving:
*   Setting up a Vector Database (e.g., Chroma).
*   Implementing an LLM integration (for RAG and interaction).
*   Developing the backend logic for conflict detection and knowledge management.
