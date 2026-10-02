# Business Marketing Content Flow

This file contains prompts for generating marketing concepts from business information.

<!-- PROMPT: business_content -->

## System

You are an expert business research and marketing strategist.

## User

Analyze the business information below and identify the company's core value propositions.

Business Name: {{business_name}}

Business Content:
{{business_content}}

Generate exactly 3 unique business value themes that would resonate with potential customers.

---

<!-- PROMPT: business_value_themes -->

## System

You are a creative advertising strategist.

## User

Based on the selected business value theme, generate 3 visually evocative
visual concepts that could be used for advertising content.

Selected Business Value Theme:
{{business_value_themes}}

Focus on:
- The target customer
- The business value proposition
- A visually compelling scene
- The desired emotional response

---

<!-- PROMPT: visual_concepts -->

## System

You are an expert AI image prompt writer and creative director.

## User

Based on the selected visual concept, create 3 detailed image-generation prompts.

Selected Visual Concept:
{{visual_concepts}}

Each prompt should include:

- Subject
- Composition
- Visual style
- Lighting
- Color and mood
- Photography or illustration direction
- Important visual details

Present the result as clean Markdown with headings and numbered sections.
Do not return JSON.

---

<!-- PROMPT: marketing_caption -->

## System

You are an expert marketing copywriter.

## User

Create a compelling marketing caption based on the selected visual concept.

Selected Visual Concept:
{{visual_concepts}}

The caption should:
- Clearly communicate the business value
- Be concise and engaging
- Appeal to potential customers
- Include a clear call to action