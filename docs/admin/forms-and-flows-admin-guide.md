# PromptFlow — Form and Flow Admin Guide

This guide explains how to prepare reusable prompt forms, compose flows, assign them to organizations, and use every setting currently declared by the form and flow JSON schemas. It distinguishes accepted JSON from behavior implemented by the runners.

**Reference version:** repository `chollins/promptflow`, application commit `d1a08a75e553284dab4608d86181f0b2de43bbfe`, reviewed October 27, 2026. This documents the repository, not a verified hosted deployment.

## Contents

- [Before you start](#before-you-start)
- [Create a form](#create-a-form)
- [Import forms from Markdown](#import-forms-from-markdown)
- [Create a flow](#create-a-flow)
- [Configure dependent steps](#configure-dependent-steps)
- [Form JSON reference](#form-json-reference)
- [Flow JSON reference](#flow-json-reference)
- [Admin API settings](#admin-api-settings)
- [Models, outputs, and execution behavior](#models-outputs-and-execution-behavior)
- [Test and troubleshoot](#test-and-troubleshoot)
- [Implementation references](#implementation-references)

## Before you start

A **form** defines input fields and the system/user prompt sent to a model. A **flow** places forms in a sequence and carries context between steps. Forms are reusable: changing one can affect every flow that uses it.

Creating/editing forms and flows, creating organizations, and assigning flow access require a **superadmin** account. An organization admin can manage its team and organization model labels, but these authoring endpoints do not grant authoring rights to ordinary organization admins.

You need a working backend/database and provider access to test real generation. OpenAI text and DALL-E paths require configured OpenAI credentials. Leonardo has a separate key and falls back to mock results without one. Never put credentials in form/flow JSON, prompts, or documentation. A model listed in a selector is not proof that its provider is integrated.

Use test information first. Generated outputs may be saved automatically. Public result-sharing links can expose inputs and output, as well as author/organization display names.

### Understand the three kinds of identifiers

| Identifier | Where it comes from | Where it is used |
| --- | --- | --- |
| Database ID | Assigned when a form/flow is saved | Admin API operations and form-to-flow relationships |
| Database slug | Generated from the outer Name, with a suffix on collisions | Runtime lookup, URLs, and generated step IDs |
| JSON `id` | Written inside Content JSON/Flow JSON | Returned runtime definition, runner requests, and context references |

For predictable lookup, keep the JSON `id` equal to the saved slug. For example, Name `Meeting Summary` normally yields slug `meeting-summary`; use `"id": "meeting-summary"`. Check the actual saved slug, especially after duplicate names or renames. A form JSON ID unrelated to its saved slug can cause subsequent runner requests to fail. Renaming metadata does not automatically rewrite IDs and references inside all definitions.

Use simple lowercase IDs. Use underscores for field IDs (`meeting_notes`) and simple stable step IDs (`summary`). Avoid spaces, commas in checkbox option values, and dots in ordinary field IDs: dots in prompt placeholders mean nested paths.

## Create a form

1. Open **Forms administration** at `/admin/forms` and choose the new-form action.
2. Enter Name and an optional Description. These are database metadata, separate from similarly named values inside Content JSON.
3. Paste a complete form definition into **Content JSON**, or use **Upload JSON** to load a file into that editor. Uploading JSON populates the editor; save the form afterward.
4. Leave **Active** checked for an intended usable form, then choose **Create form**. Active is stored metadata; it is not consistently enforced by runtime loaders, so do not use it as an access-control switch.
5. Inspect the saved slug and align the internal JSON `id` if needed.
6. Add the form to an assigned flow before testing with an ordinary organization member. Member form access is derived from forms linked to flows assigned to that organization.

### Example: Meeting Summary

Use outer Name `Meeting Summary`. This full definition asks for structured output so a later step can consume it:

```json
{
  "id": "meeting-summary",
  "name": "Meeting Summary",
  "description": "Turn supplied notes into a summary and action items.",
  "version": "1.0",
  "fields": [
    {
      "id": "meeting_notes",
      "label": "Meeting notes",
      "description": "Paste test notes without confidential information.",
      "type": "textarea",
      "required": true
    },
    {
      "id": "tone",
      "label": "Tone",
      "type": "dropdown",
      "options": ["Professional", "Concise"],
      "default": "Professional",
      "required": true
    }
  ],
  "prompt": {
    "system": "Summarize only the supplied notes. Do not invent facts. Return only a JSON object with summary (string) and actions (array of strings).",
    "user": "Use a {{tone}} tone. Notes: {{meeting_notes}}"
  },
  "output": {
    "type": "object",
    "schema": {
      "type": "object",
      "properties": {
        "summary": {"type": "string"},
        "actions": {"type": "array", "items": {"type": "string"}}
      },
      "required": ["summary", "actions"]
    }
  },
  "execution": {"mode": "interactive"}
}
```

Omitting `model` uses the form schema's defaults. The current admin save path serializes those defaults back into stored JSON, so the saved form can acquire an explicit model and override a flow fallback even when the original input omitted it. Verify the saved definition and actual runner selection. The `output.schema` describes the desired shape; it does **not** enforce that shape or automatically add instructions to the prompt. Keep the JSON-return instruction in the prompt and check actual results.

To edit, open the existing form's edit action, change metadata/JSON, and save. Back up the JSON before substantial changes. Deleting a form can affect linked flows; inspect its usage first.

## Import forms from Markdown

The forms admin page offers Markdown import/upload actions. Unlike Upload JSON, the Markdown action immediately calls the import endpoint; it does not merely fill the currently open JSON editor.

```markdown
<!-- PROMPT: meeting-summary -->
<!-- SYSTEM: Summarize supplied notes without inventing facts. -->
Summarize these notes: {{meeting_notes}}

<!-- PROMPT: follow-up-draft -->
<!-- SYSTEM: You write concise internal follow-up messages. -->
Write a follow-up about {{topic}} for {{audience}}.
```

Each `PROMPT` marker starts a form. IDs allow letters, digits, underscores, and hyphens. `SYSTEM` is optional; its default is `You are a helpful AI assistant.` Text between markers, after removing the system marker, becomes the user prompt. Text before the first marker is not a form.

The importer derives names from IDs and creates required fields from placeholders in both prompts. Fields default to text; variable names containing `copy`, `summary`, `desc`, `text`, `details`, `content`, or `prompt` become textareas. It sets version `1.0` and model `openai/gpt-4o-mini`, temperature `0.7`. Review the generated JSON to adjust types, choices, defaults, model, and output settings. Because the model is explicit, it can take precedence over a flow default.

Duplicate IDs within a file, missing markers, and empty prompt bodies are rejected. Import is an **upsert**: a matching existing form may be updated and reactivated. It is not a guaranteed copy operation, and a multi-form import is not a single all-or-nothing transaction. Back up existing definitions before reimporting. Keep placeholders simple; dotted placeholders require nested context rather than ordinary flat form values.

## Create a flow

### Choose one authoring mode

| Mode | How to author | What runs |
| --- | --- | --- |
| Linked-form composition | Add existing forms through **Flow steps** | Steps generated from the database form links |
| JSON-defined steps | Edit **Flow JSON**, leaving linked-form rows empty | The steps in the saved JSON definition |

**Critical precedence:** If a flow has any linked-form rows, the loader reconstructs its steps from those links. It does not merge those rows with JSON-defined steps. Generated steps use the form slug with hyphens replaced by underscores as step ID, the form slug as `prompt_form_id`, and the link order as `sequence`. They have empty `input_bindings`/`dynamic_fields`, no step output settings, `next: null`, and review metadata derived from the Required checkbox. The generated flow uses version `1.0`, metadata name/description/slug, the database model configuration, and default runtime settings.

Consequently, advanced step bindings, output formats, and step IDs written into Flow JSON have no effect while linked forms exist. Form-level `data_source` can still work in linked composition when it references the generated step ID.

### Compose a flow with the step editor

1. Open `/flows` as superadmin. Choose the create-flow action, enter Name/Description, and select **Create & Compose**.
2. In the editor at `/admin/flows/<database-id>`, select a form in **Flow steps** and choose **Add to Flow**. **New Form** opens a form-creation dialog; after creating it, add the selected form to the flow.
3. Add subsequent forms and drag rows to reorder them. Reordering and add/remove actions save immediately. Metadata/Flow JSON changes use the separate **Update flow** action.
4. Set the flow model provider/name/temperature if needed. The temperature slider spans 0–2 in steps of 0.1. It is a UI range, not schema/provider validation.
5. Save metadata. Before running, include at least one step; the schema accepts an empty array but the runtime needs a step.
6. Use the flow access-management action on `/flows`, or the organization's admin detail screen, to assign the flow to the intended organization.
7. Test as a member of that organization, not only as superadmin.

The same database form cannot be linked twice to one flow: `(flow_id, form_id)` is the relationship's primary key. For repeated use of one form, use distinct IDs in JSON-defined steps instead.

The link's **Required** checkbox becomes `review.required`; it does not make the form's input fields required and does not currently create an enforced approval gate or optional-step behavior. Field-level `required` is separate.

### Define advanced steps in Flow JSON

Create an empty flow, then edit Flow JSON without adding linked-form rows. Use this full definition with outer Name `Meeting Follow Up`, and confirm its saved slug is `meeting-follow-up`:

```json
{
  "id": "meeting-follow-up",
  "version": "1.0",
  "name": "Meeting Follow Up",
  "description": "Summarize notes and draft an internal follow-up.",
  "runtime": {"mode": "guided", "default_review_required": true},
  "model": {"provider": "openai", "name": "gpt-4o-mini", "temperature": 0.7},
  "steps": [
    {
      "id": "summary",
      "sequence": 1,
      "name": "Summarize meeting",
      "prompt_form_id": "meeting-summary",
      "dynamic_fields": ["meeting_notes", "tone"],
      "review": {"required": true, "editable": true},
      "output": {"save_as": "meeting_recap", "formats": ["json", "markdown"]},
      "next": "follow_up"
    },
    {
      "id": "follow_up",
      "sequence": 2,
      "name": "Draft follow-up",
      "prompt_form_id": "follow-up-draft",
      "input_bindings": {"summary_text": "meeting_recap"},
      "dynamic_fields": ["audience"],
      "review": {"required": true, "editable": true},
      "output": {"save_as": "follow_up_message", "formats": ["markdown"]},
      "next": null
    }
  ]
}
```

Create a second form with outer Name `Follow Up Draft`, saved slug/JSON ID `follow-up-draft`:

```json
{
  "id": "follow-up-draft",
  "name": "Follow Up Draft",
  "fields": [
    {"id": "audience", "label": "Audience", "type": "text", "required": true}
  ],
  "prompt": {
    "system": "Write a concise internal follow-up. Use only the supplied summary and actions.",
    "user": "Audience: {{audience}}. Previous meeting result: {{summary_text}}"
  }
}
```

The first result is stored as a raw string under `meeting_recap`. The second step's binding inserts that string as `summary_text`; it works even if the model did not return valid JSON. By contrast, dependent choice fields need correctly parsed structured output.

**Access limitation:** Ordinary members' form access is checked through linked-form relationships. Merely referencing a form in JSON steps does not create those relationships. A JSON-only flow assigned to an organization can therefore load while its forms fail access checks for members. Use JSON-only definitions for controlled superadmin testing/file workflows until this is addressed; do not assume the advanced example is ready for member distribution. Adding links changes execution precedence rather than resolving this transparently.

## Configure dependent steps

After a flow step executes, context includes:

```json
{
  "steps": {
    "summary": {
      "input": {"meeting_notes": "Test notes", "tone": "Professional"},
      "output": {"summary": "A test summary", "actions": ["Confirm the date"]},
      "raw": "Original model response text",
      "status": "completed",
      "error": null
    }
  }
}
```

This is **execution state**, not authoring JSON. To offer choices from the previous output, add a field such as:

```json
{
  "id": "selected_action",
  "label": "Action to expand",
  "type": "dropdown",
  "required": true,
  "data_source": {"type": "step_output", "step_id": "summary", "path": "actions"}
}
```

The web flow runner reads `context.steps.summary.output.actions`, expects an array, and turns entries into choice strings. `path` is relative to the step's parsed output; use `actions`, not `steps.summary.output.actions`. Nested paths can use dots. Other `data_source.type` strings are accepted by the web schema but have no implemented source resolver. Standalone form rendering does not resolve step-output choices, and the CLI does not reproduce the browser's choice UI.

In linked composition, reference the generated ID instead; for `meeting-summary`, use `meeting_summary`. To reference earlier information directly in a prompt, use `{{steps.summary.input.meeting_notes}}` or `{{steps.summary.output.summary}}`. Supply nested context, not a literal flat key containing dots.

In the **web service**, `input_bindings` uses **exact top-level context keys**, not dotted-path resolution: `"summary_text": "meeting_recap"` works; `"summary_text": "steps.summary.output.summary"` does not traverse that path. Use the dotted prompt expression directly for that case. The separate file runtime does resolve binding paths starting with `steps.`; do not assume that behavior is shared by the web service. Step-value assembly starts with submitted values, overlays context values, then overlays resolved bindings, so same-named context/bound values can override newly submitted fields.

## Form JSON reference

### General validation rules

The web definition is validated by Pydantic. “Required” below means the key must be present; it does not imply nonempty strings, unique IDs, or valid cross-references. Unknown properties are ignored rather than implemented. Use JSON booleans/numbers, not strings, even where coercion might accept them. Comments and trailing commas are not JSON.

Names, IDs, and array lengths have no explicit minimum/maximum constraints in these schemas. Keep IDs unique and steps nonempty yourself. No custom validators, file-upload fields, number fields, multiselect type, conditional-visibility expressions, or field min/max/regex rules are declared. Choice arrays must contain strings even though some frontend rendering code can display richer objects.

### Top-level form object

| Key | Type | Required / default | Meaning and current behavior |
| --- | --- | --- | --- |
| `id` | string | Required | Runtime identifier; align with the saved form slug. |
| `name` | string | Required | Runtime display name. Separate from outer admin metadata. |
| `description` | string or null | `null` | Optional runtime description. |
| `version` | string | `"1.0"` | Definition metadata; not automatic version history. |
| `fields` | array of field objects | Required | Inputs in display order; empty is accepted. |
| `prompt` | prompt object | Required | System and user instructions. |
| `model` | model object | OpenAI / GPT-4o Mini / 0.7 | Explicit form model takes priority over the flow default. Admin persistence materializes defaults even when omitted in the input. `null` is not accepted. |
| `output` | output object or null | `null` | Optional structured-output declaration; see below. |
| `execution` | execution object or null | `null` | Holds a mode label; not automatic execution scheduling. |

### `fields[]`

| Key | Type | Required / default | Meaning and current behavior |
| --- | --- | --- | --- |
| `id` | string | Required | Input key and prompt placeholder, e.g. `meeting_notes`. |
| `label` | string | Required | Visible field label. |
| `description` | string or null | `null` | Help text where the UI displays it. |
| `type` | enum string | Required | `text`, `textarea`, `date`, `checkbox`, `radio`, `dropdown`, `hidden`. |
| `required` | boolean | `false` | Used by UI gating and CLI required-input checks; not comprehensive backend input validation. |
| `default` | string or null | `null` | Initial value for most web fields. Checkbox initialization uses `"false"` rather than this default. |
| `options` | array of strings | `[]` | Choices for dropdown/radio; checkbox with choices becomes a multiple-choice list. Web schema rejects `null`. |
| `data_source` | data-source object or null | `null` | Dynamic choices from previous step output in the web flow runner. |

Field types:

| Type | Value / presentation |
| --- | --- |
| `text` | Single-line string. |
| `textarea` | Multiline string. |
| `date` | Browser date input; value is a date string. No backend date validator is declared here. |
| `radio` | One option string. |
| `dropdown` | One option string; use nonempty option strings. |
| `checkbox` with options | Selected values joined with commas, not a JSON array. Avoid commas inside individual choices. Initial `"false"` may not represent a selected option; inspect actual values. |
| `checkbox` without options | Boolean-like string: checked is `"true"`; unchecked can be `"false"` or empty, depending on runner. Required-state checks do not consistently enforce “must be checked.” |
| `hidden` | Hidden input; set an appropriate default/context source. Hidden does not mean confidential or immutable. |

### Nested form objects

| Object / key | Type | Required / default | Behavior |
| --- | --- | --- | --- |
| `prompt.system` | string | Required | System instruction; placeholders are rendered. |
| `prompt.user` | string | Required | User instruction; placeholders are rendered. |
| `model.provider` | string | `"openai"` | Metadata, not a general provider dispatcher. |
| `model.name` | string | `"gpt-4o-mini"` | Model identifier sent to execution; name-based image routing applies. |
| `model.temperature` | number | `0.7` | No schema range bound; provider/model may restrict supported values. |
| `data_source.type` | string | Required | Use `"step_output"`; other strings have no implemented web resolver. |
| `data_source.step_id` | string or null | `null` | Actual earlier runtime step ID; necessary for useful step-output resolution. |
| `data_source.path` | string or null | `null` | Dot-separated path within earlier parsed output; set it explicitly. |
| `output.type` | string | Required when `output` exists | Typically `"object"`; arbitrary strings are accepted. Not an output-handler selector. |
| `output.schema` | object or null | `null` | Descriptive JSON-shaped schema. Web model also accepts the internal spelling `schema_`; use `schema` for portability. Not enforced against model responses. |
| `execution.mode` | string | `"interactive"` when object exists | Use `"interactive"` or `"automatic"` for portability; arbitrary strings accepted on web. Neither starts unattended web execution. |

Prompt replacement recognizes `{{field_id}}` and nested dot paths, without expression evaluation, loops, filters, or formatting functions. Use braces with no internal spaces. Missing paths can leave unresolved placeholders and cause execution errors; test the actual rendered prompt. Objects/lists resolved by the renderer become JSON strings. Standalone execution converts submitted form values to strings; flow context can retain structured objects.

**Form `output` limits:** Standalone form execution returns raw text and does not invoke flow output handlers. In a flow, a truthy form `output` triggers an attempt to parse the response as a JSON object/array, stripping surrounding code fences. On failure it falls back to raw text; it does not reject schema violations. Dependent choices will fail if they need an array that is absent or unparsed.

### File-runtime differences

`backend/runtime/schemas/prompt_form.py` is used for file-based CLI/Python work. Its field options may be `null` and default to `null`; its `data_source.type` is restricted to `step_output` and `step_id`/`path` are required strings. `execution.mode` is restricted to `interactive` or `automatic`. Use non-null string-array options, fully populated data sources, and those two mode strings to stay compatible with both paths. Runtime output uses the `schema` alias; avoid relying on web-only acceptance of `schema_`.

For standalone file execution, make referenced forms available as matching `<form-id>.form.json` files under `backend/forms` or the supported sample-form locations, or use Markdown/runtime-registered forms. Saving a form in the web admin database alone does not make it available to a standalone CLI process outside a Flask application context.

## Flow JSON reference

### Top-level flow object

| Key | Type | Required / default | Meaning and current behavior |
| --- | --- | --- | --- |
| `id` | string | Required | Runtime flow ID; align with saved slug for web requests. |
| `version` | string | Required | Unlike forms, there is no version default. Metadata only. |
| `name` | string | Required | Runtime display name. |
| `description` | string | `""` | Description; `null` is not the declared type. |
| `runtime` | runtime object | Defaults below | Runtime/review metadata. `null` is not accepted. |
| `model` | model object or null | `null` | Flow fallback model; same keys/defaults as form model. |
| `steps` | array of step objects | Required | Ordered by numeric `sequence` during execution; use unique step IDs and sequence numbers. Empty is accepted by validation but not a usable web flow. |

### `runtime`

| Key | Type | Default | Meaning and current behavior |
| --- | --- | --- | --- |
| `mode` | `"guided"` or `"automatic"` | `"guided"` | Accepted mode label. Web UI still executes a step at a time. CLI file runner loops through all steps regardless of this label. |
| `default_review_required` | boolean | `true` | Stored setting; not an enforced web approval gate. |

### `steps[]`

| Key | Type | Required / default | Meaning and current behavior |
| --- | --- | --- | --- |
| `id` | string | Required | Stable context key, e.g. `summary`. Must be unique to avoid ambiguous lookup/overwriting context. |
| `sequence` | integer | Required | Numeric sort order. No positivity or uniqueness validator; use 1, 2, 3. |
| `name` | string | Required | Step label. |
| `prompt_form_id` | string | Required | Existing form lookup identifier, normally saved slug; the schema does not check existence/access. |
| `input_bindings` | object of string → string | `{}` | Prompt variable → exact top-level context key on web. File runtime additionally resolves paths beginning with `steps.`. |
| `dynamic_fields` | array of strings | `[]` | Nonempty list selects/orders fields displayed for this step; empty displays all form fields. Not conditional field creation; unknown field IDs are omitted. CLI required-input validation does not use this display filter. |
| `review` | review object | Defaults below | Metadata; flags are not enforced by the current web runner. `null` is not accepted. |
| `output` | step-output object or null | `null` | Named raw context output plus optional backend file artifacts. Different from form `output`. |
| `next` | string or null | `null` | Returned next-step metadata. Web navigation and CLI iteration use sequence order, not this pointer. It does not implement branching. |

### `steps[].review`

| Key | Type | Default | Meaning and current behavior |
| --- | --- | --- | --- |
| `required` | boolean | `true` | Intended review setting; no current enforced approval/skip behavior. |
| `editable` | boolean | `true` | Intended edit setting; does not currently restrict the intermediate raw-text editor. |

The UI can edit intermediate `stepResult`, but continuing only copies edits to a configured named output. It does not generally rewrite stored structured `context.steps` or history. Do not depend on editing JSON text to refresh downstream choice arrays. Rerun the step when structured context must change.

### `steps[].output`

| Key | Type | Required / default | Meaning and current behavior |
| --- | --- | --- | --- |
| `save_as` | string | Required | Top-level context name for raw result and base artifact filename. Use a simple unique name, not a path. |
| `formats` | array of strings | `["json"]` | Handler names: `json`, `markdown`, `leonardo`, or `image` (alias for Leonardo). Empty list writes no artifacts but the raw named context output is still set. |

`json` parses valid JSON or wraps plain text as `{"result": "..."}` and writes `<save_as>.json`. `markdown` writes raw text as `<save_as>.md`. `leonardo`/`image` can trigger image generation, not merely export an existing response. Do not include them in ordinary text flows unless another provider action is intended. Generation can cost money; an image-generating model plus an image output handler can cause an additional generation request.

Unknown format strings pass schema validation but return a failed/unregistered handler result; they do not install new capabilities. Handler failures are not necessarily surfaced as failed flow execution. JSON has no settings for handler-specific `options`, Markdown transformation templates, or download formats such as `docx`/`csv`; those are separate programmatic/browser features. Unknown extra keys will be ignored.

Backend artifacts are written below `backend/outputs/<flow-id>/` (with image artifacts in an `images` subdirectory). Reusing names can overwrite files. This is separate from the database Saved Results archive and browser exports.

## Admin API settings

These are request-envelope settings, **not** additional top-level form/flow definition properties. UI editors send `content_json` as a JSON-encoded **string**, not an embedded JSON object.

| Operation | Endpoint | Request fields |
| --- | --- | --- |
| Create/update form | POST `/api/admin/forms`; PUT `/api/admin/forms/<id-or-slug>` | `name` (string), `description` (string or null), `content_json` (string containing full form definition), `is_active` (boolean; default true) |
| Import Markdown | POST `/api/admin/forms/import-markdown` | `content` (Markdown string), or `file_path` (path readable by backend); content takes precedence. Browser imports send content, not a local computer path. |
| Create/update flow | POST `/api/admin/flows`; PUT `/api/admin/flows/<id-or-slug>` | `name`, `description`, `content_json`, `is_active`; optional `model_configuration` object using the model settings above |
| Add linked step | POST `/api/admin/flows/<id-or-slug>/steps` | `form_id` (database ID or slug), `is_required` (boolean, default true). A submitted `step_number` is ignored; backend appends after the highest number. |
| Reorder linked steps | PUT `/api/admin/flows/<id-or-slug>/steps/reorder` | `steps`: array of `{ "form_id": "...", "is_required": true }`; submitted order replaces link ordering, starting at 1. Omitted entries are removed. |
| Remove linked step | DELETE `/api/admin/flows/<flow-id>/steps/<form-id>` | No body; removal uses database relationship identifiers. |
| Assign organization access | POST `/api/admin/organizations/<organization-id>/flows` | `flow_id` (flow database ID, not slug) |
| Revoke organization access | DELETE `/api/admin/organizations/<organization-id>/flows/<flow-id>` | No body; check current assigned IDs. |

Outer name/description do not automatically synchronize all inner JSON metadata. Form slugs are recalculated on rename. Saving a flow with a nonempty `model_configuration` writes it into the flow definition's `model`; if no separate configuration is supplied, a JSON model can populate the database model configuration. In linked composition the database configuration is the fallback model.

GET `/api/admin/forms`, `/api/admin/forms/<id-or-slug>`, `/api/admin/flows`, and `/api/admin/flows/<id-or-slug>` provide administrative metadata/definitions. DELETE `/api/admin/forms/<id-or-slug>` and `/api/admin/flows/<id-or-slug>` remove records. Use the UI when possible and preserve a definition backup before destructive changes. These are authenticated superadmin operations, not public APIs for anonymous authoring.

## Models, outputs, and execution behavior

### Model resolution

Flow steps choose: **request override → explicit form model → flow model → form/system default** (`openai`, `gpt-4o-mini`, `0.7`). Web runners normally send a selected-model override, defaulting to a browser favorite or GPT-4o; consequently the form/flow default may not determine the actual model. The flow UI sends temperature `0.7` with its override. Standalone form execution uses its own form model and an optional request override; it has no flow fallback.

The admin create/update form service serializes default values, including `model`, into persisted definitions. Reloading can therefore treat a default model as explicit. The flow-fallback rule is most straightforward with file/direct definitions that truly omit `model`; omitting it before saving through the current admin UI does not guarantee the same result.

The provider string is not a general routing configuration. Current execution uses model-name matching for Leonardo/DALL-E, otherwise OpenAI chat. Catalog entries for other providers/audio do not make those integrations work. Leonardo's selected catalog model name is not transmitted as a provider model identifier in its request payload. Confirm actual supported model access with the operator.

### Execution request fields

For administrators integrating clients, POST `/api/forms/<id>/execute` accepts `values` (object) and `model_override` (model-name string or object); `model` and `model_name` are also accepted aliases. Form overrides use name/model and optional temperature; provider is not dispatched.

POST `/api/flows/<id>/execute` accepts `values`, `context` (prior returned context), `step_id` (omitted selects first sorted step), and `model_override` (model object), with `model` as alias. The normal client should supply a complete model object. This endpoint executes **one** step and returns updated context; it does not persist a resumable workflow session for you.

With auto-save enabled, successful forms and final flow-step outputs are submitted to Saved Results; intermediate steps offer manual Save Result. Saving may fail independently of generation, and matching inputs/output may update an existing saved result. Check the archive rather than assuming every run is retained as a version.

## Test and troubleshoot

Use this checklist before assigning a new workflow:

1. Validate JSON syntax and schema. Then verify referenced forms exist; schema parsing does not resolve them.
2. Confirm JSON IDs, saved slugs, runtime step IDs, and dependent-field references match.
3. Choose linked composition or JSON-defined steps deliberately. Inspect GET `/api/flows/<id>` to see the effective runtime definition.
4. Fill required inputs and use harmless test data. Superadmins may see rendered prompts/diagnostics depending on deployment policy; ordinary members may not.
5. Before Execute, confirm provider configuration and possible costs. Check the chosen model override rather than only the saved flow default.
6. Inspect raw and formatted results, structured outputs, dependent choices, next-step inputs, final output, exports, and saved results. Do not equate valid JSON with correct content.
7. Test organization-member access separately, including direct form URLs. Advanced JSON-only references currently have the access limitation described above.
8. Revisit an earlier step and rerun it to check downstream choices reset appropriately. A manual text edit does not reliably refresh structured context.

| Symptom | Likely cause / action |
| --- | --- |
| JSON saves but a setting has no effect | Unknown keys may be ignored; review metadata-only settings and linked-step precedence. |
| Form not found / 403 | Wrong internal ID/slug, missing referenced form, unassigned organization, or JSON-only form-access limitation. |
| Flow appears empty | Empty `steps` array or no linked forms. Populate the chosen mode. |
| Unresolved prompt variables | Typo, missing context, a dot-path used as a flat field, or a binding incorrectly expecting nested lookup. |
| Loading/unavailable dynamic choices | Referenced step has not executed, wrong generated step ID/path, invalid JSON output, or non-array value. |
| Flow default model ignored | Runner request override or explicit form model takes precedence. |
| Provider error / mock image | Missing credentials, unavailable model, unsupported integration, or Leonardo mock fallback. |
| Inactive item remains runnable | Active flags are not reliable runtime enforcement; use actual organization access controls. |
| Required checkbox not checked but execution enabled | Checkbox string representation and required gating are inconsistent; validate important inputs independently. |
| Changes to JSON steps do not appear | Linked-form rows replace those steps. Do not combine modes expecting a merge. |
| Result is displayed but absent from history/files | Auto-save or output handler may fail separately. Inspect logs/permissions and verify the intended save mechanism. |

Local schema/example validation does not verify a deployment or live model behavior. Do not advertise unattended automation, enforced approvals, schema-guaranteed output, or full model-provider coverage based on the accepted JSON settings.

## Implementation references

- Web form schema: `backend/services/schemas/prompt_form.py` (`PromptForm`, `FieldSchema`, `DataSource`, `ModelSettings`, `OutputSchema`, `ExecutionSettings`).
- Web/file flow schemas: `backend/services/schemas/prompt_flow.py`, `backend/runtime/schemas/prompt_flow.py` (`PromptFlow`, `FlowStep`, `RuntimeSettings`, `ReviewSettings`, `OutputSettings`).
- File form schema: `backend/runtime/schemas/prompt_form.py` (stricter source/mode declarations).
- Loading and persistence: `backend/services/form_service.py` (`create_form`, `update_form`, `import_forms_from_markdown`); `backend/services/flow_service.py` (`load_flow_definition`, `create_flow`, `update_flow`).
- Admin/access endpoints: `backend/routes.py` (`admin_form_create`, `admin_flow_create`, `admin_flow_add_step`, `admin_flow_reorder_steps`, `_require_form_access`, `admin_organization_add_flow`).
- Editors: `frontend/src/routes/admin.forms.tsx`, `frontend/src/routes/flows.tsx`, `frontend/src/routes/admin.flows_.$id.tsx`.
- Execution and context: `backend/services/form_executor.py`, `flow_executor.py`, `context_service.py`, `prompt_service.py`, `prompt_executor.py`; `backend/file_runner.py` (sequential file execution).
- Field/runner behavior: `frontend/src/components/FieldRenderer.tsx`, `DynamicForm.tsx`, `form-runner.tsx`, `flow-runner.tsx`; `frontend/src/lib/favorite-models.ts`.
- Markdown parser: `backend/services/markdown_prompt_parser.py` (`parse_markdown_prompts`).
- Output handlers: `backend/services/output_handlers/registry.py`, `json_handler.py`, `markdown_handler.py`, `leonardo_handler.py`; image request fields belong to `backend/services/adapters/leonardo_adapter.py`, not to the form/flow schema.
- Existing focused tests: `backend/tests/test_markdown_parser.py`, `test_flow_model_resolution.py`, `test_cli_execution.py`, `test_output_handlers.py`, `test_leonardo_handler.py`.
