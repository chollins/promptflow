Yes. Since this is for your developers, I’d make the README **practical and copy-pasteable**, with the setup order:

1. Prerequisites
2. Backend
3. Frontend
4. Database
5. Environment variables
6. Run the application
7. Verify installation
8. Common troubleshooting

I’d also explicitly document that **`jericca` contains the environment/configuration file**, so developers know where to get it rather than accidentally committing secrets.

# PromptFlow — Local Development Setup

This document explains how to set up PromptFlow locally for development.

## 1. Prerequisites

Install the following:

* Python 3.x
* pip
* Node.js + npm
* XAMPP
* Git

Recommended:

* VS Code
* MySQL/MariaDB client
* Postman or similar API testing tool

---

# 2. Project Structure

The project is divided into:

```text
project/
├── backend/
├── frontend/
└── README.md
```

The backend is a Flask application.

The frontend is a Node/npm application.

The database runs through MySQL/MariaDB provided by XAMPP.

---

# 3. Backend Setup

## 3.1 Open the backend directory

```bash
cd backend
```

## 3.2 Create a Python virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

If activation succeeds, the terminal should show something similar to:

```text
(venv)
```

---

## 3.3 Install Python dependencies

Install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

If `pip` is outdated:

```bash
python -m pip install --upgrade pip
```

Then:

```bash
pip install -r requirements.txt
```

Do not manually install individual packages unless a dependency is intentionally being added to the project.

When adding a new Python dependency, update:

```text
requirements.txt
```

---

# 4. Environment Configuration

The project requires environment variables for local development.

The development environment file is provided through the `jericca` environment/configuration source.

IMPORTANT:

* Do not commit `.env` files containing secrets.
* Do not copy API keys directly into Python source code.
* Do not hardcode database passwords.
* Ask for/access the project's approved `jericca` environment configuration when setting up a new machine.

The backend should load configuration from environment variables.

Typical configuration includes:

```text
DATABASE_URL
SECRET_KEY
OPENAI_API_KEY
```

Use the project's existing configuration names. Do not rename environment variables unless the application configuration is updated accordingly.

---

# 5. Database Setup

PromptFlow currently uses MySQL/MariaDB.

XAMPP is used for local database development.

## 5.1 Start XAMPP

Open XAMPP Control Panel.

Start:

```text
Apache
MySQL
```

The important service for PromptFlow is:

```text
MySQL
```

Apache is only required if another part of the local environment depends on it.

---

## 5.2 Create the database

Open:

```text
http://localhost/phpmyadmin
```

Create the development database configured by the project environment.

Example:

```text
promptflow
```

Use the database name configured in the project's environment variables if it differs.

Do not manually create application tables.

The tables should be created through migrations.

---

# 6. Database Migrations

Make sure the backend virtual environment is activated:

```bash
venv\Scripts\activate
```

From the backend directory, use the project's Flask migration commands.

Check that Flask recognizes the application:

```bash
flask --app app.py
```

If the project uses Flask-Migrate, initialize/apply migrations according to the existing migration setup.

For an existing migration repository, apply migrations with:

```bash
flask --app app.py db upgrade
```

This creates/updates the database schema based on the migration files.

### Creating a new migration

After changing SQLAlchemy models:

```bash
flask --app app.py db migrate -m "Describe the schema change"
```

Then apply it:

```bash
flask --app app.py db upgrade
```

Do not manually modify tables in phpMyAdmin when the change belongs to the application's schema.

---

# 7. Database Seeder

After migrations have been applied, run the project seed process.

Example:

```bash
python seed.py
```

The seeder is responsible for creating development/sample data such as:

* roles
* users
* organizations
* flows
* forms
* flow/form relationships
* sample configuration

Use the project's current `seed.py` implementation.

If the seeder is designed to be idempotent, it should be safe to run again without creating duplicate records.

If seed data has changed, rerun:

```bash
python seed.py
```

Do not manually insert required application records through phpMyAdmin unless specifically debugging the database.

---

# 8. Recommended Database Setup Order

For a new developer machine:

```text
1. Start XAMPP
       ↓
2. Start MySQL
       ↓
3. Create development database
       ↓
4. Configure environment variables
       ↓
5. Activate Python virtual environment
       ↓
6. pip install -r requirements.txt
       ↓
7. flask db upgrade
       ↓
8. python seed.py
```

---

# 9. Frontend Setup

Open a second terminal.

Navigate to the frontend:

```bash
cd frontend
```

Install npm dependencies:

```bash
npm install
```

This installs dependencies defined in:

```text
package.json
```

If the project uses a lockfile, keep the lockfile committed.

Do not delete or regenerate the lockfile unnecessarily.

---

# 10. Start the Backend

From:

```text
backend/
```

activate the virtual environment:

```bash
venv\Scripts\activate
```

Start Flask using the project's configured command.

Typical development command:

```bash
flask --app app.py run
```

The API should then be available on the configured local backend address.

Example:

```text
http://127.0.0.1:5000
```

Use the actual port configured by the project if different.

---

# 11. Start the Frontend

From:

```text
frontend/
```

run:

```bash
npm run dev
```

Vite will display the local frontend URL in the terminal.

Example:

```text
http://localhost:5173
```

Open the displayed URL in the browser.

---

# 12. Development Workflow

Normally run three components:

```text
┌───────────────────────┐
│ XAMPP                 │
│ MySQL                 │
└───────────┬───────────┘
            │
            ↓
┌───────────────────────┐
│ Flask Backend         │
│ API + Flow Runner     │
└───────────┬───────────┘
            │
            ↓
┌───────────────────────┐
│ Frontend              │
│ React/Vite            │
└───────────────────────┘
```

Recommended terminal setup:

### Terminal 1 — Database

Start XAMPP:

```text
MySQL: ON
```

### Terminal 2 — Backend

```bash
cd backend
venv\Scripts\activate
flask --app app.py run
```

### Terminal 3 — Frontend

```bash
cd frontend
npm run dev
```

---

# 13. First-Time Setup Checklist

* [ ] Install Python
* [ ] Install Node.js/npm
* [ ] Install XAMPP
* [ ] Clone the repository
* [ ] Obtain the approved `jericca` environment configuration
* [ ] Create/activate Python virtual environment
* [ ] Install backend dependencies
* [ ] Start MySQL through XAMPP
* [ ] Create the development database
* [ ] Apply database migrations
* [ ] Run the database seeder
* [ ] Run `npm install`
* [ ] Start Flask
* [ ] Start frontend
* [ ] Open the application
* [ ] Verify login/authentication
* [ ] Verify database-backed Flow data loads
* [ ] Verify a Flow can be executed

---

# 14. Verifying the Installation

After starting the application, verify:

## Backend

* Flask starts without errors.
* Database connection succeeds.
* API endpoints respond.
* No missing environment-variable errors occur.

## Database

Verify that expected application tables exist.

Important: PromptFlow is database-driven.

The database is the source of truth for:

```text
Flows
Forms
Flow/Form relationships
Users
Organizations
Permissions
Execution/runtime data
```

Do not expect Forms or Flows to be loaded from JSON files.

## Frontend

Verify:

* frontend starts successfully
* API requests reach Flask
* authentication works
* flows load
* forms load
* Flow Runner works

---

# 15. Important Architecture Note

PromptFlow does NOT use a JSON filesystem architecture.

There are no:

```text
forms/*.json
flows/*.json
```

that the application relies on at runtime.

Instead:

```text
Database
   │
   ├── Flow
   │
   ├── flow_form_steps
   │
   └── Form
          │
          └── configuration/content
```

JSON may still be used as structured data inside database fields, but the database is the source of truth.

---

# 16. Database Changes During Development

When modifying SQLAlchemy models:

1. Modify the model.
2. Generate a migration.

```bash
flask --app app.py db migrate -m "Describe change"
```

3. Review the generated migration.
4. Apply it.

```bash
flask --app app.py db upgrade
```

5. Update the seeder if the new schema requires development data.
6. Test the application.

Never assume that changing a Python model automatically changes the existing database.

---

# 17. Resetting Local Development Data

If the local development database needs to be recreated, coordinate with the developer/team before deleting data.

A full reset generally means:

```text
Drop development database
       ↓
Create database again
       ↓
Run migrations
       ↓
Run seed.py
```

Do NOT reset a shared/staging/production database using local development procedures.

---

# 18. Common Problems

## `ModuleNotFoundError`

Example:

```text
ModuleNotFoundError: No module named 'flask_cors'
```

Make sure the virtual environment is active:

```bash
venv\Scripts\activate
```

Then:

```bash
pip install -r requirements.txt
```

---

## `flask db` command does not exist

Check that:

* Flask-Migrate is installed.
* The application is correctly configured.
* The migration extension is initialized.
* The correct Flask app is being loaded.

Try:

```bash
flask --app app.py
```

and verify the available commands.

---

## Database connection error

Check:

1. XAMPP MySQL is running.
2. Database exists.
3. Database credentials are correct.
4. Environment variables are loaded.
5. Database host/port match the local MySQL configuration.

---

## Tables do not exist

Run:

```bash
flask --app app.py db upgrade
```

Then run the seeder if required:

```bash
python seed.py
```

---

## Duplicate seed data

If the seeder reports duplicate records, determine whether the seeder is expected to be idempotent.

Do not blindly delete database records.

Check the existing seed logic first.

---

## Frontend cannot connect to backend

Verify:

1. Flask is running.
2. Frontend API configuration points to the correct backend URL.
3. CORS/configuration is correct.
4. Browser developer tools → Network shows the request.
5. Flask terminal shows the incoming request.

---

# 19. Before Committing Changes

Backend:

```bash
pip freeze
```

Only update `requirements.txt` when intentionally changing dependencies.

Frontend:

```bash
npm install
```

Ensure `package-lock.json` is updated when dependencies change.

Database:

* Include migration files.
* Do not commit local database files/dumps containing sensitive data.
* Do not commit secrets.

Environment:

* Never commit API keys/passwords/secrets.
* Use the approved environment configuration.

---

# 20. Quick Start

Once the machine has been configured previously:

### 1. Start XAMPP

```text
MySQL → ON
```

### 2. Backend

```bash
cd backend
venv\Scripts\activate
flask --app app.py run
```

### 3. Frontend

```bash
cd frontend
npm run dev
```

Then open the frontend URL displayed by Vite.

If database migrations are required because the code has changed:

```bash
cd backend
venv\Scripts\activate
flask --app app.py db upgrade
```

If new development seed data is required:

```bash
python seed.py
```
