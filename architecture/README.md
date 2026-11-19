# KAIROS-AI architecture sheet

This folder contains the repo-grounded architecture diagram in an A3 landscape format. Revision 02 presents three abstraction levels on one sheet:

- high level: system context and future hardware boundary;
- mid level: the four-stage runtime path;
- low level: concrete modules, data contracts and persisted outputs.

- `a3.html` is the browser preview and print source.
- `a3.pdf` is the print-ready A3 landscape export.
- `architecture.css` and `tokens.css` carry the Hallmark Console direction.

The diagram reflects the current prototype in `app.py` and `modules/`. The future ESP32 / MQTT / HTTP path is intentionally shown as a dashed boundary because the README says it is not connected.
