# Role-based diagnostic visibility

**Status:** Proposed

**Audience:** Backend, frontend, security, and QA engineers

## Summary

PromptFlow currently decides whether to return execution debug data with a hard-coded
`superadmin` role check. This change replaces that decision with explicit,
server-owned configuration for each supported role. The same effective policy is
returned to the frontend so that it can render only the diagnostic sections the
server permits.

The server remains the security boundary: hidden data must be omitted from API
responses, not merely hidden with CSS or React conditionals.

## Goals

- Allow operators to independently configure diagnostic visibility for
  `superadmin`, `admin`, and `user` accounts.
- Apply one policy consistently to form and flow executions.
- Control diagnostic categories independently, including JSON, prompt text, model
  configuration, input provenance, raw model output, and timing/runtime details.
- Preserve the current default behavior: superadmins receive all existing debug
  information; admins and users receive none.
- Give the UI enough capability information to avoid offering unavailable panels.
- Fail closed when configuration or role data is invalid.

## Non-goals

- Changing authorization for forms, flows, organizations, or admin pages.
- Controlling application logs, persisted output files, browser developer tools,
  or error-reporting integrations.
- Redacting individual secrets inside a permitted diagnostic category. Redaction is
  a separate defense and should still be applied before any diagnostic value is
  serialized.
- Allowing users to override their own policy in a request.

## Terminology and roles

The product may call the highest-privilege account a **superuser**, but the current
database role is `superadmin`. This specification uses `superadmin` as the canonical
configuration and API key. UI copy may continue to say “superuser.” No separate
`superuser` role is introduced.

Only these normalized role names are recognized:

1. `superadmin`
2. `admin`
3. `user`

Missing roles, unknown roles, unauthenticated callers, malformed policies, and
unrecognized diagnostic category names resolve to no diagnostic access.

## Configuration contract

Add the following Flask configuration values. Each value is a comma-separated list
of category identifiers.

| Flask setting | Environment variable | Default |
| --- | --- | --- |
| `DIAGNOSTICS_SUPERADMIN` | `DIAGNOSTICS_SUPERADMIN` | `all` |
| `DIAGNOSTICS_ADMIN` | `DIAGNOSTICS_ADMIN` | `none` |
| `DIAGNOSTICS_USER` | `DIAGNOSTICS_USER` | `none` |

Parsing rules:

- Values are case-insensitive; trim whitespace around tokens.
- `none` must appear alone and grants no categories.
- `all` must appear alone and expands to every category known by that release.
- Otherwise, every token must be one of the category identifiers below.
- Empty values, duplicate tokens, mixed sentinel values (for example
  `all,prompts`), and unknown tokens are invalid.
- Production startup must fail with a clear configuration error if any value is
  invalid. Tests and development use the same validation to prevent environments
  from behaving differently.
- Defaults are applied only when an environment variable is absent, not when it is
  present but empty.

Example:

```dotenv
DIAGNOSTICS_SUPERADMIN=all
DIAGNOSTICS_ADMIN=prompts,structured_output,execution
DIAGNOSTICS_USER=none
```

### Diagnostic categories

| Identifier | Response content controlled | Frontend presentation |
| --- | --- | --- |
| `input_sources` | `debug.input_sources`, including bound values and paths | Input Sources |
| `prompts` | `debug.prompt_template`, `debug.resolved_prompt`, and the existing `steps[].prompt` field | Prompt Template and Resolved Prompt |
| `model` | `debug.model_configuration` | Model Configuration |
| `output_schema` | `debug.output_schema` | Output Schema |
| `raw_response` | `debug.raw_response` | Raw Response |
| `structured_output` | A parsed/pretty-printed JSON view derived from the ordinary step result | JSON / Structured Output |
| `execution` | `debug.execution_details` and `debug.runtime_state` | Execution Details and Runtime State |

`structured_output` controls only the additional diagnostic JSON viewer. It does
not suppress the normal user-facing result required by the workflow. Likewise,
`raw_response` is a duplicate diagnostic representation and does not authorize
access to a result that the caller could not otherwise receive.

The category registry must be centralized in the backend. Adding a new diagnostic
field in the future requires assigning it to a category and adding tests proving
that it is absent without that category. Because `all` is forward-expanding,
operators who require a fixed allowlist should enumerate categories instead.

## Effective-policy resolution

Introduce a backend policy service with one entry point conceptually equivalent to:

```python
diagnostic_policy_for(user) -> frozenset[DiagnosticCategory]
```

The service normalizes the authenticated user's database role, selects the matching
validated setting, and returns an immutable category set. Route handlers must not
contain role-specific diagnostic checks.

The policy is an allowlist. Execution code may avoid collecting expensive details
when their category is disabled, but a response serializer/filter must still apply
the allowlist immediately before serialization. This second step prevents a newly
collected field from leaking accidentally.

Authorization and diagnostic visibility are independent. A policy never grants
access to a form or flow; it is evaluated only after the existing resource-access
check succeeds.

## API changes

### Execution endpoints

Apply the policy to both:

- `POST /api/forms/{form_id}/execute`
- `POST /api/flows/{flow_id}/execute`

Successful responses gain a `diagnostic_capabilities` array containing the
effective category identifiers in stable alphabetical order. The array is always
present for authenticated execution responses and may be empty.

```json
{
  "context": {},
  "steps": [{"id": "draft", "result": "..."}],
  "debug": {
    "prompt_template": {"system": "...", "user": "..."},
    "resolved_prompt": {"system": "...", "user": "..."},
    "execution_details": {"duration_ms": 218}
  },
  "diagnostic_capabilities": ["execution", "prompts"]
}
```

Response rules:

- Omit `debug` when no permitted debug fields remain. Do not return `debug: null`.
- Within `debug`, omit disabled keys rather than returning `null`, empty objects, or
  redacted placeholders.
- Omit `steps[].prompt` unless `prompts` is permitted. This closes the existing path
  that exposes the rendered user prompt outside `debug`.
- Do not accept `include_debug`, a category list, or a role in the request body or
  query string. Such input cannot expand the server-derived policy.
- Error responses must not include diagnostics, stack traces, prompt text, input
  values, or the policy configuration.
- Form and flow execution must use identical category-to-field mappings.

### Session capability discovery

Add `diagnostic_capabilities` to the existing authenticated session/user response
so the frontend can build the initial page without guessing from the role. This is
advisory only; execution responses remain authoritative in case configuration
changes during a session.

Do not expose the full role-to-policy map to clients. A caller learns only their own
effective capabilities.

## Frontend behavior

- Replace role-name checks for diagnostic panels with capability checks.
- Render a diagnostic panel only when at least one returned capability has content.
- Render each tab/section only when its category is present and its response field
  exists. Never reconstruct prompt or raw-response diagnostics from unrelated
  response fields as a fallback.
- Show the structured JSON viewer only for `structured_output`; continue to show the
  ordinary workflow result regardless of that capability.
- Treat an absent or malformed capability list as an empty list (fail closed).
- Clear the previous execution's diagnostic state before starting another request
  and on logout/account changes so data permitted to one user cannot remain visible
  to the next user.
- Do not persist diagnostic payloads in local storage or client analytics.

## Security and privacy requirements

- Server-side filtering is mandatory even if the frontend panel is hidden.
- Filtering must occur before `jsonify`, logging of response bodies, caching, or
  analytics instrumentation.
- Existing secret-redaction rules apply to all enabled categories. At minimum,
  recursively redact values whose keys match `authorization`, `api_key`, `token`,
  `password`, `secret`, or `cookie` (case-insensitive) before serialization.
- Diagnostic responses must retain the existing `Cache-Control` behavior for
  authenticated data; if none is set today, add `Cache-Control: no-store` to
  execution responses.
- Configuration changes require an application restart for the first
  implementation. Dynamic per-request configuration is out of scope.

## Observability

At startup, log the enabled category names for each role, but never log diagnostic
values. For each execution, structured logs may include the user ID, normalized
role, and enabled category names. Do not log prompts, input values, raw responses,
session credentials, or environment-variable contents.

## Implementation sequence

1. Define the category enum/registry, configuration defaults, parser, validation,
   and policy resolver.
2. Change form and flow executors to accept a category set instead of the
   `include_debug` boolean, and make prompt fields optional.
3. Add a final response-filtering layer and capability metadata to both execution
   endpoints.
4. Expose the current user's effective capabilities through session discovery.
5. Update frontend response types and diagnostic components to use capabilities.
6. Document the environment variables in `backend/.env.example` and deployment
   configuration before release.

## Test plan

### Configuration tests

- Verify defaults and parsing of whitespace/case.
- Verify `all` expansion and explicit allowlists.
- Verify empty, unknown, duplicated, and mixed-sentinel values fail startup.

### Backend policy and API tests

- Parameterize every role against every category and assert exact response keys.
- Confirm the default matrix matches current intended access: all for
  `superadmin`, none for `admin` and `user`.
- Confirm `prompts` independently controls templates, resolved prompts, and
  `steps[].prompt`.
- Confirm disabled categories cannot be enabled through request parameters.
- Confirm unknown/missing roles and unauthenticated requests fail closed.
- Confirm access checks still reject unauthorized forms/flows regardless of policy.
- Confirm secret-like keys are recursively redacted and responses use `no-store`.
- Snapshot form and flow response shapes to ensure category parity.

### Frontend tests

- Render each diagnostic section from its capability independently.
- Verify no diagnostic panel appears for an empty or malformed capability list.
- Verify stale diagnostic state is cleared between executions and users.
- Verify structured JSON visibility does not hide the ordinary workflow result.

### End-to-end acceptance criteria

1. With defaults, superadmins retain all current diagnostic views, while admins and
   users receive no diagnostic fields or rendered prompts in the network response.
2. Enabling only `prompts` for admins makes prompt tabs available to admins without
   exposing input sources, model settings, raw output, JSON diagnostics, or timing.
3. Enabling only `structured_output` for users shows the JSON diagnostic view when
   applicable without exposing raw response or prompts.
4. A policy change takes effect after restart and is reflected both in session
   discovery and subsequent execution responses.
5. Browser inspection confirms that hidden diagnostic values are absent from the
   payload, not merely hidden in the interface.

## Rollout and compatibility

The defaults preserve the existing debug allowlist, but omitting `steps[].prompt`
for admins and users is an intentional security correction and may affect clients
that relied on that undocumented field. Announce the response-shape change and
provide one release of notice if external clients consume it.

Roll out first with explicit production values matching the defaults, monitor
execution errors and payload sizes, and then enable narrower categories for admins
or users as required. Rollback consists of restoring the previous environment
values and restarting; it must not require a database migration.
