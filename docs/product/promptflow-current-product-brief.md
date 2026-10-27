# PromptFlow — Current Product Brief

**Purpose:** A factual reference for Digital Mixology product positioning and website content. Prepared for an initial **product preview**, not a declaration of public availability.

**Reviewed:** October 27, 2026 (America/Los_Angeles). Repository: `chollins/promptflow`; checked-out branch: `work`; commit: `d1a08a75e553284dab4608d86181f0b2de43bbfe`. The checkout was clean before review. Only this brief was added; no application changes affect these findings. No deployed application was reviewed.

## Product summary

PromptFlow organizes reusable AI prompts into forms and ordered workflows for organizations. A user supplies information through defined fields, selects a model, runs a prompt or workflow step, and receives an AI-generated result. Results can be reviewed, exported, saved, and optionally shared by link. Platform administrators maintain the prompt library and assign workflows to organizations. [E1–E7]

The problem it addresses is the repeated work of composing prompts, collecting the same information, and moving through related AI tasks without a common structure. Its current value is a reusable interface around prompts: defined inputs, a repeatable sequence of steps, and tools for retaining and distributing outputs. These mechanisms can support more consistent working practices; this review establishes no measured productivity improvement or guarantee of output quality.

**Documented intent:** The application's landing page describes building AI workflows for organizations, inviting teams, and sharing prompts. The README describes model configuration, Markdown prompt imports, command-line execution, and output handlers. The implementation plan describes intended architecture; it is not treated as proof of completion. Digital Mixology ownership and the product-preview objective are supplied by the product owner for this review. [E1, E12]

**Interpretation:** Likely users include marketing and sales teams, consultants, operations staff, and internal teams that repeatedly turn structured information into drafts or summaries. A smaller technical audience can prepare templates and run file-based workflows from Python or a terminal. This is an interpretation of the implemented tools and examples, not evidence of customer adoption.

## Current user experience

### Access and team setup

The practical entry point is an existing account or an administrator-issued invitation. An invited person opens a token-bearing registration link, sets a name and password, and then signs in. Organization administrators can invite and manage members; platform superadministrators create organizations and assign their workflow access. Invitation creation returns a registration link, but successful email delivery depends on a configured Mailjet integration. Password recovery also depends on email. [E2]

A public organization-creation screen exists, but its submission targets an unsupported API operation. A local route check returned HTTP 405. The landing page's “Get started” link is commented out. These are material reasons not to promote open self-service signup. [E2, E13]

### Run one reusable prompt

A signed-in user opens an available form, fills in its fields, chooses a model, and selects Execute. Forms support text, longer text, dates, checkboxes, radio choices, dropdowns, and hidden values. The backend inserts the supplied values into stored prompt instructions and calls the execution service. The result appears in a formatted view; the user can execute again, export it, or select Save Result. Actual AI generation requires working provider access. [E3, E8]

The setup must contain a usable form and, for ordinary members, organization access through an assigned flow. Example files in the repository do not establish that those examples are assigned or available in a deployed account.

### Run a guided workflow

A flow presents its forms as ordered steps. The user supplies inputs, executes the current step, reviews the result, and continues. Execution context carries earlier inputs and outputs forward; configured fields can offer choices derived from an earlier structured response. The interface includes previous-step navigation and step history. [E4]

This is a guided, browser-managed sequence. The web endpoint executes one step per request; schema fields named “automatic” do not establish unattended web automation. Intermediate raw text can be edited, but propagation is limited: continuing updates a named output when one is configured, while stored structured step output and history are not generally synchronized with the edit. Do not describe this as a complete approval or revision system. [E4]

### Retain and distribute the work

With auto-save enabled, successful form results and the final flow-step result are submitted to Saved Results. Users can also save manually, search results, filter by form or flow, inspect inputs and available JSON, copy output, and delete accessible records. Administrators' visibility depends on their role and configuration. Repeated matching inputs or output can update an existing record rather than preserve every run as a separate version. Auto-save errors are caught separately from execution, so successful generation does not guarantee successful archival. [E5]

Runner exports offer Word, Markdown, and CSV. Sharing a saved result enables a public token-based view; disabling sharing makes that view unavailable. The public response includes the output, input summary, and author and organization display names. A shared flow link is different: it points to a workflow that still requires account access. [E6, E7]

### Maintain the templates

Superadministrators can create and edit forms using JSON, import specially marked Markdown prompts, arrange reusable forms into flows, reorder steps, configure flow defaults, and assign flows to organizations. Organization administrators can give models organization-specific labels. This is implemented administration, but template creation still requires understanding prompt definitions and structured configuration. It should not be sold as a comprehensive no-code workflow designer for every team member. [E9]

## Capability and availability inventory

**Implementation labels describe this commit, not release status.** “Repository only” means code was found; no deployed observation establishes availability. Local tests are reported separately below. “Partially implemented” identifies a material gap within the named capability.

| Capability | What users can do | Implementation status | Deployment verification | Constraints or dependencies | Evidence |
| --- | --- | --- | --- | --- | --- |
| Accounts and invitations | Sign in, accept invitations, change or recover passwords | Implemented | Repository only | Existing account/invitation; email configuration for delivery | E2 |
| Public workspace signup | Use the organization-creation screen | Partially implemented | Repository only | Submission calls unsupported POST operation | E2, E13 |
| Organization access and roles | Assign flows; manage members and role-scoped results | Implemented | Repository only | Different superadmin, admin, and member responsibilities; not a security certification | E2, E5, E9 |
| Reusable prompt forms | Supply defined inputs and execute a stored prompt | Implemented | Repository only | Accessible template and functioning AI provider | E3, E8 |
| Guided flows and dependent choices | Run ordered steps and reuse prior context | Implemented | Repository only | Definition and structured response must match; browser holds run state | E4 |
| Editing and unattended web runs | Edit intermediate text; definitions expose runtime flags | Partially implemented | Repository only | Edit propagation is limited; no full automatic web-run loop found | E4 |
| Form/flow authoring | Import Markdown, edit JSON, add and reorder forms | Implemented | Repository only | Superadmin permissions; technical template preparation | E9 |
| Model selection and labels | Choose models, favorite them, and set organization labels | Implemented | Repository only | Favorites are browser-local; catalog presence does not prove execution support | E8, E10 |
| OpenAI text generation | Generate text through the OpenAI client | Implemented | Repository only | Credentials, compatible model, network access and provider availability; no live call tested | E8 |
| Image generation | Request DALL-E or Leonardo images and display results | Partially implemented | Repository only | Live fulfillment unverified; Leonardo has mock fallback and model-selection limitations | E8, E11 |
| Other provider/audio catalog entries | Browse listed text, image, and audio models | Placeholder or mockup | Repository only | Catalog metadata exists; corresponding general provider/audio execution adapters were not found | E8, E10 |
| Saved Results | Save, search, inspect, copy, and delete accessible outputs | Implemented | Repository only | Configurable auto-save and admin access; deduplication can replace earlier content | E5 |
| Public result links | Enable or disable a public saved-result view | Implemented | Repository only | Anyone holding an enabled link can view the included information | E6 |
| File exports | Download runner output as DOCX, Markdown, or CSV | Implemented | Repository only | Browser conversion; CSV is table/text conversion, not a general data integration | E7 |
| CLI and Python execution | Run local definitions with dictionary/JSON inputs and optional JSON output | Implemented | Repository only | Technical workflow and provider access; execution tests substitute AI responses | E12 |
| Dashboard health panel | See “All Operational” indicators | Placeholder or mockup | Repository only | Static labels; recent activity is browser-local, not a central audit trail | E13 |

## Representative use cases

These are supported scenarios, not claims that a particular deployed account contains the templates. Text generation remains conditional on working OpenAI access; outputs need human review.

1. **Turn meeting notes into a useful draft.** A team member enters a title, date, participants, notes, and desired format. The meeting-summary example asks for an executive summary, action items, or detailed minutes. The result is a draft that can be retained or exported; there is no meeting recording or transcription integration established here. [E14]
2. **Prepare repeatable sales outreach.** A seller provides prospect and company context, the outreach goal, tone, and length. The sales-email example generates an email draft. It does not send that draft or synchronize it with a CRM. [E14]
3. **Reuse a marketing prompt across a team.** A template owner imports the supplied social-media Markdown example. Users provide topic, platform, audience, message, and tone, then receive a draft post. Sharing the template through assigned workflows avoids requiring every user to write the underlying instructions. No social-network publishing integration was found. [E9, E14]
4. **Prepare image concepts — depends on incomplete/unverified integration.** A configured image prompt can reach DALL-E or Leonardo and return image metadata for display. This should be presented as a preview capability until live fulfillment is verified. Without its key, Leonardo produces mock metadata, not a real image. No claim of successful commercial image production follows from the passing mock tests. [E11]

## Integrations and material limitations

**AI services.** Text execution uses OpenAI through LangChain. Image branches use OpenAI's image API and a Leonardo adapter. The executor routes by model name; the general provider field does not implement interchangeable Anthropic, Google, Stability, or ElevenLabs clients. Audio models in the catalog do not establish speech generation or transcription. The Leonardo request payload also does not transmit the selected catalog model identifier. Confirm the exact supported models before publishing logos or compatibility lists. [E8, E10, E11]

**Formats and operating dependencies.** Definitions are JSON; Markdown imports require explicit prompt markers and variable placeholders. Browser exports support DOCX/Markdown/CSV; backend output handlers support JSON, Markdown, and Leonardo/image processing. Python and CLI execution use local files and a separate runtime path, without the web Saved Results archive. The web app depends on a Flask backend and database. Documentation names MySQL in setup instructions and PostgreSQL in the implementation plan, while configuration defaults to SQLite. Several migrations use `now()` defaults that need compatibility handling on SQLite. Deployment database compatibility remains unverified. [E7, E9, E12, E15]

**Availability and governance.** No deployment URL or login environment was identified in the reviewed material. No hosted workflow, real AI request, email delivery, browser download, or screenshot was verified. Public result sharing can disclose inputs as well as outputs. Access checks and diagnostic filtering exist, but this review does not establish comprehensive tenant isolation, privacy guarantees, compliance, retention controls, performance, or reliability. Landing-page Privacy and Terms links are placeholders. [E2, E6, E13]

**Workflow boundaries.** No scheduler, durable paused-run recovery, collaborative live editing, general connector marketplace, or billing workflow was established in the reviewed implementation. In the sample Client Assessment flow, a prior summary is bound into context, but the supplied sales-email prompt does not reference that summary; avoid claiming this example demonstrates summary-informed outreach. Passing tests do not cover every browser interaction or integration. [E4, E14, E16]

## Website positioning recommendations

The following is recommended copy, separate from verified implementation facts.

**Concise description:** “PromptFlow is a Digital Mixology product for turning reusable AI prompts into guided workflows for teams, with structured inputs and tools to review, save, and share results.”

**Supportable benefits:**

- Give repeat tasks a consistent starting point with reusable prompt forms.
- Guide people through related AI tasks one step at a time.
- Put prepared prompts within reach of team members through organization-assigned workflows.
- Keep useful outputs accessible through Saved Results and document exports.
- Share selected results by link, with the ability to disable sharing.

These benefits describe available mechanisms, not measured outcomes. Lead the preview with a simple form, an ordered flow, and the saved/exported result. Present administration and model labels as supporting organizational capabilities. Keep image generation qualified and omit unimplemented provider/audio claims.

Avoid “available now,” “sign up free,” “fully automated,” “works with every AI model,” “secure/private by default,” “complete audit history,” guaranteed accuracy, quantified savings, or enterprise-readiness claims. Do not imply that ordinary members can author everything. Do not infer a beta, waitlist, pricing, or public enrollment program from the code.

**Recommended CTA:** “Discuss PromptFlow with Digital Mixology,” linked to an existing, confirmed contact route. This is a recommendation to establish an appropriate contact action; no working contact, enrollment, or demo-booking mechanism was verified in this repository.

## Evidence, freshness, and owner decisions

This brief describes the commit recorded above, reviewed October 27, 2026. Repository URL: https://github.com/chollins/promptflow. The checkout is on `work`, although the original environment request selected `main`; no equivalence to the current remote branch tip is asserted. There were no pre-existing uncommitted changes. No deployed URL/environment was reviewed, and no production availability is inferred.

Existing dependencies were reused. All **88 backend tests passed** with 12 warnings; the frontend production build passed. Type checking failed with four component-prop errors, and lint reported 982 errors and 16 warnings. These results support local implementation review, not production operation. Detailed commands and limits are in E16. No dependencies were installed, application code edited, deployment performed, or commit created.

Before website copy is finalized, the product owner should confirm:

1. Who can access the preview today, and which deployed URL and version represent it?
2. Which workflows, models, and image integrations are intentionally enabled and have been exercised live?
3. Who authors templates, provisions organizations, and supplies/pays for provider access?
4. Is public sharing of inputs and display names intended, and what data-use, retention, privacy, and legal guidance is approved?
5. What CTA and real contact destination should the product preview use?

## Evidence appendix

References below are repository-relative paths with starting line numbers and relevant symbols. They identify implementation evidence at the reviewed commit. URLs appearing in integration entries are destinations encoded in the implementation, **not independently verified online sources**. No external product claims were sourced online.

| ID | Evidence and significance |
| --- | --- |
| E1 | `frontend/src/routes/index.tsx` (`Landing`, organizational workflow copy); `frontend/src/app.tsx` (`App`, actual React Router routes). `frontend/src/routes/README.md` describes a different routing framework and is not treated as current implementation evidence. |
| E2 | `backend/routes.py:295` (`_get_current_user`), `:332`–`:390` (role/access helpers), `:420` (`auth_login`), `:482`–`:630` (password changes/recovery), `:1312` (`admin_create_organization`), `:1488`–`:1566` (invitation validation/acceptance), `:1620`–`:1826` (invitations and users). `frontend/src/routes/signup.tsx` requires a token; `frontend/src/routes/create-organization.tsx:65` submits POST `/api/organizations`, for which no POST handler exists. `backend/services/email_service.py:32` (`send_email`), `:91` (`send_password_reset_email`); encoded email destination: https://api.mailjet.com/v3.1/send. |
| E3 | `backend/services/schemas/prompt_form.py:25` (`FieldSchema`), `:47` (`PromptForm`); `backend/services/form_executor.py:37` (`execute_form`); `backend/routes.py:646`–`:713` (catalog and form access), `:886` (execution). `frontend/src/components/form-runner.tsx:55` (`FormRunner`, execution, selection, export, save actions); `frontend/src/components/DynamicForm.tsx` and `FieldRenderer.tsx` (input presentation). |
| E4 | `backend/services/flow_executor.py:162` (`_execute_step`), `:329` (`execute_flow`, single-step response); `backend/services/context_service.py:9` (`ExecutionContext`); `backend/routes.py:960` (`run_flow`). `frontend/src/components/flow-runner.tsx:284` (`FlowRunner`, React state), `:419` (dependent fields), `:493` (`runCurrentStep`), `:531` (`continueToNext`), `:726` (raw edit), `:870` (history). Edits change `stepResult`; they do not rewrite `stepResults` or `context.steps` generally. `backend/services/flow_service.py:45` maps database-linked forms into steps with empty bindings and no named output; advanced flow-definition settings therefore do not automatically carry into every database-composed flow. |
| E5 | `backend/routes.py:886` and `:960` (form/final-step auto-save and separate save-error handling), `:1849`–`:1918` (list/manual save/detail/delete). `backend/services/saved_result_service.py:11` (`save_execution_result`, matching-input/output updates), `:70` (`get_user_access_filter`); `backend/config.py` (save/admin-access defaults). `frontend/src/routes/saved-results.tsx` (search, filters, detail tabs). `backend/tests/test_saved_results.py:56` (`TestSavedResultsPermissions`) checks role hierarchy, deduplication, deletion and sharing. |
| E6 | `backend/services/saved_result_service.py:147` (`share_saved_result`), `:165` (`unshare_saved_result`); `backend/routes.py:1919`–`:1961` (sharing/public response, including inputs and display names). `frontend/src/routes/saved-results.tsx` (`handleOpenShare` enables sharing); `frontend/src/routes/shared-result.tsx` (public view). `frontend/src/routes/flows.tsx:718` (Share Flow dialog) copies a `/flows/...` URL, subject to backend flow access checks. |
| E7 | `frontend/src/components/export-dropdown.tsx` (`ExportDropdown`); `frontend/src/lib/export-utils.ts:21` (`exportToMarkdown`), `:36` (`exportToCsv`), `:91` (`exportToDocx`). `backend/services/output_handlers/registry.py` (`OutputHandlerRegistry`), `json_handler.py`, `markdown_handler.py`, `leonardo_handler.py`; `backend/tests/test_output_handlers.py` covers handler behavior, not browser download fidelity. |
| E8 | `backend/services/prompt_executor.py:119` (`execute_prompt`): Leonardo-name branch, OpenAI key requirement, DALL-E branch, then `ChatOpenAI`; no general provider dispatcher. `backend/runtime/prompt_executor.py:119` has the parallel file-runtime path. `backend/services/flow_executor.py:21` (`resolve_effective_model`) resolves request override, explicit form model, flow model, then default. Runner model selection sends an override; frontend defaults/favorites can supersede flow defaults. OpenAI calls use SDK destinations; no live request was made. |
| E9 | `backend/routes.py:774`–`:843` (superadmin form CRUD/Markdown import), `:1076`–`:1210` (flow CRUD and steps), `:1261`–`:1310` (organization assignments), `:185`–`:292` (organization model labels). `backend/services/markdown_prompt_parser.py:20` (`parse_markdown_prompts`) recognizes `<!-- PROMPT: ... -->` markers and variables; `backend/services/form_service.py:180` imports forms. `frontend/src/routes/admin.forms.tsx:131` calls import; `frontend/src/routes/admin.flows_.$id.tsx` implements metadata, JSON, add/remove/reorder controls. |
| E10 | `backend/services/model_registry.py` (`DEFAULT_MODEL_CONFIG`, `seed_model_registry`, `get_available_models`), `backend/config/models.json` (catalog metadata, including unsupported providers/audio). Database seeding deactivates non-OpenAI entries; file fallback and frontend fallback lists can differ. `frontend/src/lib/favorite-models.ts:9`–`:118` stores favorites in localStorage and defaults to GPT-4o without favorites. `backend/services/organization_model_label_service.py` and `backend/tests/test_organization_model_labels.py` establish editable organization labels and tested role restrictions. |
| E11 | `backend/services/adapters/leonardo_adapter.py:27` (`LeonardoAdapter`), `:42` (`generate_image`): missing/mock-key branch, dimensions, request payload, bounded polling, artifact download. The payload omits a selected model identifier. Encoded destinations: https://cloud.leonardo.ai/api/rest/v1/generations and its generation-ID resource. `backend/services/prompt_executor.py:156` onward implements OpenAI image requests. `frontend/src/components/image-output-view.tsx` identifies mock images. `backend/tests/test_leonardo_handler.py:30` verifies mock generation; it does not verify real provider fulfillment. |
| E12 | `backend/file_runner.py:68` (`run_flow_from_file`, sequential execution and optional JSON file); `backend/cli.py` (`main`, interactive input and overrides); root `promptflow.py` and `backend/promptflow/__init__.py` (`run`); `backend/runtime/` contains file-based services. `backend/tests/test_cli_execution.py:12`, `:53`, `:117`, `:145`, `:172` exercise file/CLI/Python/Markdown paths with substituted model responses. `README.md:728` onward documents features; `implementation_plan.md` is intent, including its claim of avoiding duplicate engines, whereas separate runtime/service modules exist. |
| E13 | `frontend/src/routes/dashboard.tsx:166` and surrounding static service labels; `frontend/src/lib/activity.ts:10`–`:36` stores ten recent activities in localStorage. `frontend/src/routes/index.tsx` contains a commented registration link and `:62`–`:63` contains `#` Privacy/Terms targets. Local Flask test client confirmed unsupported public organization creation and authentication requirements; see E16. |
| E14 | `backend/sample_forms/meeting_summary.form.json`, `sales_email.form.json`; `backend/prompts/socmed.md` (topic/platform/audience/message/tone); `backend/sample_forms/photo_generator.form.json` and `backend/prompts/photo_generator.md` (image example). `backend/flows/client_assessment.flow.json` binds a recap as `summary`, but `sales_email.form.json` does not use `{{summary}}`. `backend/seed.py:277` onward supplies a guided choice-dependent example; seed/template existence does not establish deployment availability or suitability for a customer's task. |
| E15 | `backend/config.py:14` (environment-selected database URI, SQLite default), `backend/requirements.txt`, `frontend/package.json`, `frontend/vite.config.ts` (local API proxy). `README.md:130` onward names MySQL/MariaDB; `implementation_plan.md` names PostgreSQL. Migrations `8c6988cb3953_add_password_reset_otps.py`, `eb8cba18c5ee_add_saved_result.py`, `cea3f9190e5d_add_ai_models_table.py`, and `1a2b3c4d5e6f_add_organization_model_labels.py` contain `now()` defaults. Tests create in-memory tables from models, so their passing does not establish fresh migration compatibility. |

### E16 — Checks performed for this review

All commands used existing installed tools/dependencies. Generated build files and logs were directed outside the checkout. No paid API call, email, live-user mutation, or deployment was triggered.

| Check | Working directory and command | Result | What it establishes / limits |
| --- | --- | --- | --- |
| Backend suite | `backend`: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:.. /workspace/.venvs/promptflow/bin/python -m pytest tests -q -p no:cacheprovider` | Exit 0; **88 passed**, 12 warnings | Tests include saved-result permissions/sharing, labels, model resolution, parsing, mocked CLI/image behavior, diagnostics, and OTP timezone handling. In-memory/test-file fixtures are not a production database or live integration test. |
| Frontend build | `frontend`: `npm run build -- --outDir /tmp/promptflow-review-build` | Exit 0; production bundle produced; large-chunk warning | Bundling succeeds. Vite build does not imply TypeScript correctness, browser functionality, or deployment success. |
| Type checking | `frontend`: `./node_modules/.bin/tsc --noEmit` | Exit 2; **four TS2322 errors** | Invalid Badge props at `src/routes/flows.tsx:437`, `organization.tsx:310`, `shared-result.tsx:91` and `:144`. Left unchanged. |
| Lint | `frontend`: `npm run lint` | Exit 1; **982 errors, 16 warnings** | Existing formatting/code-quality findings; no automatic fixes applied. |
| Local route probes | `backend`: ephemeral Flask test client from `create_app`, `TESTING=True`, `sqlite:///:memory:`; no schema or account creation | POST `/api/organizations` → **405**; unauthenticated GET `/api/forms` and `/api/flows` → **401** | Confirms missing public creation operation and these authentication checks locally. Does not verify signup, login, tenant isolation, or a hosted application. |
| Repository state | `git status --short`, branch/HEAD, sanitized origin identity | Clean before review; `work`; commit recorded above; only requested brief added afterward | No application edits contribute to the findings. No fetch, commit, or branch update was performed. |

**Verification gaps:** No deployment URL was supplied or found in reviewed application documentation/configuration references, and no browser inspection tool was available. No deployed version comparison, hosted login, screenshots, browser exports, real AI generation, email delivery, full browser end-to-end run, production database migration, load test, or security/privacy audit was performed. These gaps do not prevent a repository-based preview brief, but they prevent release and production-readiness claims.
