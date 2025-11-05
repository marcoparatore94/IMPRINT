# IMPRINT Pilot Project

Questo repository contiene il codice per l’analisi dei marker infiammatori sistemici (SIMs) e la costruzione del modello di rischio IMPRINT.

## Struttura
- `IMPRINT_a1.0.py` → script principale
- `app/` → contiene `app.py` (interfaccia) e `risk_cli.py` (CLI)
- `model/` → script di training del modello
- `utils/` → funzioni di supporto (es. esportazione PDF)
- `data/` → dati di esempio (anonimizzati o sintetici)

## Installazione
1. Clona il repository:
   ```bash
   git clone https://github.com/<tuo-utente>/IMPRINT.git
   cd IMPRINT