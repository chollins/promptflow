# Promptflow Use Cases for a Marketing Agency

These prototype scenarios follow **Northstar Creative**, a fictional full-service agency. Together they demonstrate how Promptflow can collect structured briefs, chain AI-assisted forms, carry context forward, and produce reviewable deliverables.

## Summary comparison

| # | Use case | Agency workflow | Forms | Distinctive capabilities |
|---|---|---|---:|---|
| 1 | Campaign Brief to Concepts | Strategy and creative ideation | 3 | Guided flow, context binding, dynamic choices, editable review |
| 2 | Social Calendar Sprint | Social content planning | 3 | Structured outputs, multi-select, model tuning, Markdown/JSON export |
| 3 | Lead Nurture Email Builder | Lifecycle email production | 2 | Hidden context, defaults, automatic execution, reusable forms |
| 4 | Pitch Discovery Synthesizer | New-business discovery | 2 | Required validation, low-temperature synthesis, human approval |
| 5 | Brand Voice Rewrite Studio | On-brand copy adaptation | 1 | Standalone form, radio controls, prompt templating, rapid iteration |

## 1. Campaign Brief to Concepts

**Description:** Turn a client campaign brief into strategically aligned creative territories, then let the account team select and refine a concept.

**Flow outline:** Campaign Intake → Strategy Synthesis → Concept Selection and Refinement.

### Form fields

| Form | Key fields and types |
|---|---|
| Campaign Intake | Client name (`text`, required); campaign objective (`dropdown`); target audience (`textarea`); channels (`checkbox`, multi-select); budget band (`radio`); constraints (`textarea`) |
| Strategy Synthesis | Brief ID (`hidden`); success metric (`dropdown`); strategic emphasis (`radio`); account-team notes (`textarea`) |
| Concept Selection and Refinement | Creative territories (`radio`, dynamically populated from strategy output); desired executions (`checkbox`, multi-select); revision direction (`textarea`) |

### Features showcased

- Guided, multi-form execution with an editable review gate after strategy synthesis.
- Input bindings that preserve client, audience, channel, and constraint context across steps.
- Structured AI output used as the data source for downstream creative-territory options.
- Required-field validation and JSON plus Markdown deliverable output.

## 2. Social Calendar Sprint

**Description:** Build a channel-specific monthly content calendar from brand priorities, generated content pillars, and a strategist’s chosen post mix.

**Flow outline:** Monthly Priorities → Pillar Curation → Calendar Generation.

### Form fields

| Form | Key fields and types |
|---|---|
| Monthly Priorities | Brand (`dropdown`); month and year (`text`); business priorities (`textarea`); active channels (`checkbox`, multi-select); posting cadence (`radio`) |
| Pillar Curation | Generated pillars (`checkbox`, dynamically populated); mandatory campaign (`text`); exclusions (`textarea`); include reactive slots (`checkbox`) |
| Calendar Generation | Selected pillars (`hidden`); voice (`dropdown`); call-to-action style (`radio`); hashtag guidance (`textarea`) |

### Features showcased

- AI-generated structured pillar arrays feeding a downstream multi-select field.
- Different model temperatures: lower for pillar taxonomy, higher for post ideation.
- Step-output references and hidden fields for carrying approved context without re-entry.
- Editable intermediate review and final Markdown/JSON exports.

## 3. Lead Nurture Email Builder

**Description:** Convert a demand-generation offer and audience profile into a sequenced nurture campaign with consistent positioning and calls to action.

**Flow outline:** Nurture Setup → Email Sequence Generation.

### Form fields

| Form | Key fields and types |
|---|---|
| Nurture Setup | Offer name (`text`, required); funnel stage (`dropdown`); persona and pain points (`textarea`); sequence length (`radio`); proof points (`textarea`) |
| Email Sequence Generation | Approved positioning (`hidden`); sender name (`text`, default value); CTA (`text`); compliance statements (`textarea`); include subject variants (`checkbox`) |

### Features showcased

- Automatic flow mode for a fast handoff from setup to generation.
- Hidden cross-step bindings and default values that reduce repeated data entry.
- Reusable generation form that can accept inputs from other demand-generation flows.
- Schema-constrained output for ordered emails, subjects, preview text, body copy, and CTA.

## 4. Pitch Discovery Synthesizer

**Description:** Transform scattered prospect discovery notes into a concise opportunity assessment and a reviewable pitch-meeting agenda.

**Flow outline:** Discovery Capture → Pitch Narrative Review.

### Form fields

| Form | Key fields and types |
|---|---|
| Discovery Capture | Prospect (`text`, required); industry (`dropdown`); stakeholder notes (`textarea`, required); stated challenges (`checkbox`, multi-select); incumbent status (`radio`) |
| Pitch Narrative Review | Opportunity summary (`hidden`, bound from step output); differentiator focus (`dropdown`); sensitive claims confirmed (`checkbox`); partner feedback (`textarea`) |

### Features showcased

- Required-field validation for complete discovery inputs.
- Low-temperature model settings for consistent, evidence-grounded synthesis.
- Structured opportunity output bound into a second prompt without copy and paste.
- Mandatory, editable human review before saving the pitch narrative and agenda.

## 5. Brand Voice Rewrite Studio

**Description:** Rewrite one marketing asset for a chosen brand voice, channel, and intensity while preserving required facts and calls to action.

**Flow outline:** Rewrite Request → Generated Rewrite.

### Form fields

| Form | Key fields and types |
|---|---|
| Rewrite Request | Source copy (`textarea`, required); asset type (`dropdown`); brand voice (`radio`); rewrite intensity (`radio`); must-keep phrases (`textarea`); retain CTA (`checkbox`) |

### Features showcased

- Standalone prompt form for a focused, repeatable agency utility.
- Prompt-variable interpolation across free text and controlled-choice fields.
- Required inputs, clear field descriptions, and selectable defaults.
- Interactive execution for rapid reruns with alternate voice or intensity settings.

