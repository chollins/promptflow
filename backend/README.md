# PromptFlow Backend

Flask backend scaffold for the PromptFlow app.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
flask --app app.py run
```

## Migrations

```bash
flask --app app.py db init
flask --app app.py db migrate -m "initial"
flask --app app.py db upgrade
```
