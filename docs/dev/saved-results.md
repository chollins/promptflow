# Saved Results — Developer Specification & Documentation

## Overview

The **Saved Results** feature provides an execution history and archive for LLM-generated outputs from Form and Flow executions. Saved results record the source type (`form` or `flow`), source name, executing user, organization, input parameters summary, raw output text, and optional parsed JSON output.

---

## Access Control & Role Hierarchy

Saved results enforce a strict, server-managed permission hierarchy:

| Role | Access Scope | Delete Hierarchy |
|---|---|---|
| **Regular User (`user` / `member`)** | Can view only their own saved results (`user_id == current_user.id`). | Can delete only their own saved results. |
| **Organization Admin (`admin`)** | Can view all saved results within their organization (`organization_id == current_user.organization_id`). *Configurable via `SAVED_RESULTS_ORG_ADMIN_ACCESS`.* | Can delete any saved result within their organization. |
| **Superadmin (`superadmin`)** | Can view all saved results across all organizations on the platform. | Can delete any saved result across the entire system. |

---

## Deduplication & Upsert Behavior

To avoid cluttering history with duplicate records when running the same form/flow multiple times or clicking "Save Result" repeatedly:
1. **Deduplication Check**: Before creating a new record, the system queries for an existing `SavedResult` for `(user_id, source_type, source_id)` matching the exact `output_text` or `input_summary_json`.
2. **In-Place Update**: If a match is found, the existing record's `created_at` timestamp, `output_text`, `output_json`, and `input_summary_json` are updated in place, bringing it to the top of the history list as the latest result without creating duplicate database rows.

---

## Configuration Contract

Configuration values can be configured via environment variables or backend `config.py`:

| Environment Variable | Type | Default | Description |
|---|---|---|---|
| `SAVED_RESULTS_AUTO_SAVE` | Boolean (`true`/`false`) | `true` | When enabled, successful Form and Flow executions automatically persist a `SavedResult` record. Failed executions **never** create a saved result. |
| `SAVED_RESULTS_ORG_ADMIN_ACCESS` | Boolean (`true`/`false`) | `true` | When enabled, Organization Admins can view and delete all member outputs in their organization. When disabled, Organization Admins are restricted to their own outputs for privacy. |

---

## Database Model Schema (`SavedResult`)

Table name: `saved_results`

| Column | Data Type | Nullable | Description |
|---|---|---|---|
| `id` | `VARCHAR(36)` | No | Primary Key (UUID string). |
| `user_id` | `VARCHAR(36)` | No | Foreign Key to `users.id` (`ON DELETE CASCADE`). |
| `organization_id` | `VARCHAR(36)` | Yes | Foreign Key to `organizations.id` (`ON DELETE SET NULL`). |
| `source_type` | `VARCHAR(50)` | No | Discriminator: `"form"` or `"flow"`. |
| `source_id` | `VARCHAR(255)` | No | Identifier of the originating form or flow. |
| `source_name` | `VARCHAR(255)` | No | Human-readable name of the form or flow at execution time. |
| `input_summary_json` | `TEXT` | Yes | JSON string summarizing user inputs and parameters. |
| `output_text` | `TEXT` | No | Primary LLM output text content. |
| `output_json` | `TEXT` | Yes | Optional JSON string if output is structured. |
| `created_at` | `DATETIME` | No | Auto-populated timestamp (UTC). |
| `updated_at` | `DATETIME` | No | Auto-populated timestamp (UTC). |

---

## API Endpoint Contract

All endpoints require authentication via `X-Session-Token` or session cookie.

### 1. `GET /api/saved-results`
Lists saved results accessible to the requesting user based on role hierarchy and configuration.

**Query Parameters:**
- `source_type` *(optional)*: Filter by `"form"` or `"flow"`.
- `search` *(optional)*: Case-insensitive search on `source_name` and `output_text`.

**Response Body (200 OK):**
```json
{
  "items": [
    {
      "id": "uuid-string",
      "user_id": "user-uuid",
      "user_name": "Jane Doe",
      "organization_id": "org-uuid",
      "organization_name": "Acme Corp",
      "source_type": "form",
      "source_id": "form-123",
      "source_name": "Customer Feedback Analyzer",
      "input_summary": { "feedback": "Great service!" },
      "output_text": "### Summary\nCustomer expressed satisfaction.",
      "output_json": null,
      "created_at": "2026-09-03T20:30:00Z"
    }
  ],
  "count": 1,
  "config": {
    "auto_save": true,
    "org_admin_access": true
  }
}
```

### 2. `POST /api/saved-results`
Explicitly saves an execution result. Used when `SAVED_RESULTS_AUTO_SAVE=false` or when manually saving an output from the UI.

**Request Body:**
```json
{
  "source_type": "form",
  "source_id": "form-123",
  "source_name": "Customer Feedback Analyzer",
  "input_summary": { "feedback": "Great service!" },
  "output_text": "### Summary\nCustomer expressed satisfaction.",
  "output_json": null
}
```

**Response Body (201 Created):** Returns the serialized `SavedResult` object.

### 3. `GET /api/saved-results/<id>`
Fetches the detailed record of a single saved result. Returns `404 Not Found` if the record does not exist or the requesting user lacks access permission.

### 4. `DELETE /api/saved-results/<id>`
Deletes a saved result record. Adheres strictly to the role access hierarchy. Returns `200 OK` (`{"ok": true}`) on success or `404 Not Found` if forbidden/missing.

---

## Frontend Architecture & UI

- **Sidebar Integration**: Added to [`app-shell.tsx`](file:///c:/Projects/promptflow/frontend/src/components/app-shell.tsx) under route `/saved-results`.
- **Page Component**: [`SavedResultsPage`](file:///c:/Projects/promptflow/frontend/src/routes/saved-results.tsx) provides:
  - Filtering by type (`All`, `Forms`, `Flows`).
  - Search bar filtering results by title or output content.
  - Detail inspection modal (`Dialog`) featuring:
    - **Formatted Output (Markdown)**: Uses `react-markdown` to render formatted headings, lists, tables, and text.
    - **Structured JSON Tab**: Pretty-printed JSON view when `output_json` is present.
    - **Input Summary Inspector**: Key-value summary of inputs used.
    - **Copy to Clipboard**: Quick copy action with toast notification.
- **Manual Save Actions**: Integrated into [`form-runner.tsx`](file:///c:/Projects/promptflow/frontend/src/components/form-runner.tsx) and [`flow-runner.tsx`](file:///c:/Projects/promptflow/frontend/src/components/flow-runner.tsx) allowing direct manual output archiving.
