# app_dynamic.py
"""
Dynamic NiceGUI app driven by JSON config.

Requirements:
pip install requests beautifulsoup4 langchain langchain-openai openai nicegui


"""

import os
import asyncio
import json
import re
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from nicegui import ui
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# --------------------------
# CONFIG JSON (dynamic form)
# --------------------------
# You can reuse / adapt this CONFIG for other apps.
CONFIG_JSON = {
    "title": "AI Image Prompt Generator (Dynamic)",
    "subtitle": "Enter a business URL, let the model summarize, propose value themes, and generate visual concepts + image prompts.",
    "sections": [
        {
            "title": "1) Business Site",
            "fields": [
                {
                    "id": "business_site_url",
                    "label": "Business Site URL",
                    "type": "text",
                    "placeholder": "https://example.com",
                    "help": "Enter the business homepage URL. This triggers scraping."
                }
            ]
        },
        {
            "title": "2) Business Summary",
            "fields": [
                {
                    "id": "business_summary",
                    "label": "Business Summary",
                    "type": "textarea",
                    "placeholder": "(auto-generated)",
                    "prompt": "Generate a concise structured business summary for {business_name} based on the scraped site content below. Focus on: operations, unique value propositions, industry.\n\nScraped Content:\n{scraped_text}\n\nStructured summary:",
                    "depends_on": ["scraped_text"],
                    "readonly": True
                }
            ]
        },
        {
            "title": "3) Business Value Themes",
            "fields": [
                {
                    "id": "business_value_themes",
                    "label": "Business Value Themes",
                    "type": "button_list",
                    "prompt": "Based on the business summary below, list 3 unique value themes (short phrases) that would resonate with potential customers. Return them as newline-separated list.\n\nBusiness Summary:\n{business_summary}",
                    "depends_on": ["business_summary"]
                }
            ]
        },
        {
            "title": "4) Visual Concepts",
            "fields": [
                {
                    "id": "visual_concepts",
                    "label": "Visual Concepts",
                    "type": "button_list",
                    "prompt": "Based on the selected business value theme below, list 3 visually evocative ad concepts (short phrases). Return them as newline-separated list.\n\nSelected Theme:\n{business_value_themes}",
                    "depends_on": ["business_value_themes"]
                }
            ]
        },
        {
            "title": "5) Image Prompts",
            "fields": [
                {
                    "id": "image_prompts",
                    "label": "Image Prompts",
                    "type": "textarea",
                    "placeholder": "(auto-generated)",
                    "prompt": "Based on the selected visual concept below, write 3 specific detailed image prompts suitable for an AI image generator. Include style, subject, composition, lighting, and photography/illustration directions. Return as newline-separated list.\n\nSelected Visual Concept:\n{visual_concepts}",
                    "depends_on": ["visual_concepts"],
                    "readonly": True
                }
            ]
        }
    ]
}

# --------------------------
# HTTP + LLM setup
# --------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY environment variable")

# LangChain ChatOpenAI client (sync generate)
llm = ChatOpenAI(temperature=0, model="gpt-4")


# requests session with retries
session = requests.Session()
session.headers.update({"User-Agent": "Dynamic-Form-Scraper/1.0"})
retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://", HTTPAdapter(max_retries=retries))

# keywords (used if you want to expand link discovery)
TARGET_KEYWORDS = [
    "about", "team", "mission", "values", "services", "solutions", "products",
    "industries", "clients", "case-studies", "projects", "blog", "insights",
    "resources", "news", "careers", "jobs", "contact"
]

# --------------------------
# Utilities
# --------------------------
def parse_list_response(text: str) -> list:
    """Parse LLM list outputs into a Python list (strip numbering/bullets)."""
    if not text:
        return []
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # strip bullets or numbering like "1. ", "- ", "• "
        line = re.sub(r'^\s*[\-\*\u2022\d\.\)\:]+\s*', '', line)
        if line:
            lines.append(line)
    if not lines:
        # fallback: split by commas
        parts = [p.strip() for p in re.split(r'[,\n]+', text) if p.strip()]
        lines = parts
    # dedupe while preserving order
    seen = set()
    out = []
    for l in lines:
        kl = l.lower()
        if kl not in seen:
            out.append(l)
            seen.add(kl)
    return out[:20]

def extract_llm_text(generation_result) -> str:
    """Safely extract text from langchain-openai generate response."""
    try:
        return generation_result.generations[0][0].text
    except Exception:
        return str(generation_result)

# --------------------------
# LLM call helper
# --------------------------
async def call_llm_text(prompt_text: str) -> str:
    """Call ChatOpenAI safely and return generated text (robust extraction)."""
    loop = asyncio.get_running_loop()

    # call the llm in a thread so we don't block the event loop
    try:
        resp = await loop.run_in_executor(
            None,
            lambda: llm([HumanMessage(content=prompt_text)])
        )
    except Exception as e:
        err = f"LLM call failed: {e}"
        print("[DEBUG] call_llm_text error:", err)
        return err

    # extract text from a variety of possible response shapes
    text = ""
    try:
        # easiest cases first
        if resp is None:
            text = ""
        elif isinstance(resp, str):
            text = resp
        # ChatOpenAI sometimes returns an AIMessage-like object with .content
        elif hasattr(resp, "content"):
            text = resp.content
        # older/langchain llm.generate style LLMResult -> resp.generations[0][0].text
        elif hasattr(resp, "generations"):
            try:
                text = resp.generations[0][0].text
            except Exception:
                text = str(resp)
        # openai-like choices
        elif hasattr(resp, "choices"):
            try:
                choice = resp.choices[0]
                # choice might contain .message or .text
                if hasattr(choice, "message") and isinstance(choice.message, dict):
                    text = choice.message.get("content") or choice.message.get("text") or str(choice.message)
                else:
                    text = getattr(choice, "text", str(choice))
            except Exception:
                text = str(resp)
        else:
            # fallback to stringifying the response
            text = str(resp)
    except Exception as e:
        text = f"LLM extract error: {e}"

    # debug: print what we received
    try:
        print(f"[DEBUG] call_llm_text returned (len={len(text)}): {repr(text)[:1000]}")
    except Exception:
        print("[DEBUG] call_llm_text returned (unable to print length)")

    return text

# --------------------------
# Scraping helpers
# --------------------------
def scrape_text_from_url(url: str) -> str:
    """Basic page text extraction using BeautifulSoup."""
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
    except Exception:
        return ""
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

def discover_relevant_links(root_url: str, homepage_html: str = None):
    """Find internal links and rank by keyword presence (basic)."""
    try:
        if homepage_html:
            soup = BeautifulSoup(homepage_html, "html.parser")
        else:
            r = session.get(root_url, timeout=10); r.raise_for_status(); soup = BeautifulSoup(r.text, "html.parser")
    except Exception:
        return {}
    all_links = [urljoin(root_url, a['href']) for a in soup.find_all('a', href=True)]
    internal = [l for l in all_links if urlparse(l).netloc == urlparse(root_url).netloc]
    unique = list(dict.fromkeys(internal))
    def score(u):
        ul = u.lower()
        return sum(kw in ul for kw in TARGET_KEYWORDS)
    scored = sorted(unique, key=score, reverse=True)
    results = {}
    for kw in TARGET_KEYWORDS:
        for u in scored:
            if kw in u.lower():
                if kw not in results:
                    results[kw] = u
    return results

# --------------------------
# App dynamic engine state
# --------------------------
# Stores field values (selected values or text)
state_values = {}
# Stores generated options for button_list fields: field_id -> [options]
state_options = {}
# Widget references: field_id -> widget or container
widgets = {}
# Dependency map: field_id -> list of fields that depend on it
dependency_map = {}

# Build dependency map from config
for sec in CONFIG_JSON["sections"]:
    for fld in sec["fields"]:
        for dep in fld.get("depends_on", []):
            dependency_map.setdefault(dep, []).append(fld["id"])

# Helper: safe formatter using state_values + state_options
def build_prompt_from_template(template: str) -> str:
    """
    Format the prompt template with values from state_values and state_options.
    If an expected key is missing, substitute an empty string.
    """
    # build a dict that includes both values and options (options joined by newline)
    fmt = {}
    fmt.update({k: (v if isinstance(v, str) else ("\n".join(v) if v else "")) for k, v in state_values.items()})
    fmt.update({k: ("\n".join(v) if isinstance(v, list) else str(v)) for k, v in state_options.items()})
    # Also include 'scraped_text' if present in state_values
    try:
        return template.format(**fmt)
    except Exception:
        # fallback: safe replace braces to avoid crash
        return template

# --------------------------
# UI rendering from CONFIG
# --------------------------
ui.label(CONFIG_JSON.get("title", "Dynamic App"))
ui.label(CONFIG_JSON.get("subtitle", ""))

status = ui.label("Ready")




# Containers for sections (so UI looks grouped)
section_containers = {}

def make_on_change(_fid):
    async def _handler(event):
        # Prefer to read the authoritative widget value if widget exists
        widget = widgets.get(_fid)
        val = None

        if widget is not None:
            # ui.input widgets expose current value via .value or .get_value()
            try:
                # modern NiceGUI: widget.value
                val = getattr(widget, "value", None)
            except Exception:
                val = None

            # fallback: some widget types implement get_value()
            if val is None:
                try:
                    val = widget.get_value()
                except Exception:
                    val = None

        # If we couldn't get value from widget, fall back to event extraction
        if val is None:
            if hasattr(event, "value"):
                val = event.value
            elif hasattr(event, "args") and event.args:
                val = event.args[0]
            elif isinstance(event, str):
                val = event
            else:
                try:
                    val = str(event)
                except Exception:
                    val = ""

        # Normalise value to string and trim
        val = val or ""
        if isinstance(val, (list, dict)):
            try:
                val = json.dumps(val, ensure_ascii=False)
            except Exception:
                val = str(val)
        if isinstance(val, str):
            val = val.strip()

        # Store the value (no auto-resolve here; Next will trigger resolution)
        state_values[_fid] = val
        status.set_text(f"Set {_fid}: {val[:60]}")  # show first 60 chars for UX
        print("EVENT RAW:", repr(event))
        print("WIDGET VALUE:", getattr(widget, "value", None))

    return _handler


for section in CONFIG_JSON["sections"]:
    sec_title = section.get("title", "")
    display = "block" if sec_title.startswith("1)") else "none"
    container = ui.card().props("flat").style("margin:8px 0; padding:12px; width:100%; display:{display}")
    section_containers[sec_title] = container
    
    with container:
        ui.markdown(f"### {sec_title}")

        # build fields inside this section
        for fld in section["fields"]:
            fid = fld["id"]
            ftype = fld.get("type", "text")
            label = fld.get("label", fid)
            placeholder = fld.get("placeholder", "")
            # create widgets according to type
            if ftype == "text":
                w = ui.input(label=label, placeholder=placeholder, value="")
                # when user changes value, trigger dependency resolution
                widgets[fid] = w
                w.on('change', make_on_change(fid))
                

            elif ftype == "textarea":
                # Create textarea and store it in widgets so we can update it later
                w = ui.textarea(label=label, placeholder=placeholder, value="")
                widgets[fid] = w   # <<< IMPORTANT: keep reference so resolve_dependents_for can update it

                # If config requests readonly, set via props (best-effort)
                if fld.get("readonly"):
                    try:
                        w.props("readonly")
                    except Exception:
                        pass

                # If config specifies rows (int) or height (string), apply safely
                if "rows" in fld and isinstance(fld["rows"], int):
                    try:
                        w.props(f"rows={fld['rows']}")
                    except Exception:
                        pass
                if "height" in fld and isinstance(fld["height"], str):
                    w.style(f"height:{fld['height']}")

                # make it auto-grow instead of scroll
                w.props("autogrow")

                if fld.get("readonly"):
                    w.props("readonly")

                w.style("width:100%")
                widgets[fid] = w
            elif ftype == "button_list":
                # create a column that will hold stacked buttons
                col = ui.column().style("width:100%")
                widgets[fid] = col
                # if config provides static options, render them now
                static_opts = fld.get("options", [])
                if static_opts:
                    # create buttons stacked full width
                    for opt in static_opts:
                        def make_btn_handler(_fid=fid, _opt=opt):
                            async def _on_click():
                                state_values[_fid] = _opt
                                status.set_text(f"Selected {_opt} for {_fid}")
                                await resolve_dependents_for(_fid)
                            return _on_click
                        with col:
                            ui.button(opt, on_click=make_btn_handler()).style("width:100%; margin-bottom:6px")
            else:
                widgets[fid] = ui.label(f"Unknown field type: {ftype}")

# --------------------------
# Step control & single Next button (create AFTER all sections are built)
# --------------------------
# ordered containers in the same order as CONFIG_JSON
ordered_containers = [section_containers[sec.get("title")] for sec in CONFIG_JSON["sections"]]

# step state
_step = {"i": 0}

def show_step(index: int):
    """Make all steps up to index visible (stacked UI)."""
    total = len(ordered_containers)
    if index < 0:
        index = 0
    if index >= total:
        index = total - 1
    _step["i"] = index

    # 🔑 Instead of hiding old steps, keep them visible
    for idx, container in enumerate(ordered_containers):
        try:
            if idx <= index:
                container.style("display:block")
            else:
                container.style("display:none")
        except Exception:
            pass

    # Update button label
    try:
        if index == total - 1:
            next_button.set_text("Finish")
        else:
            next_button.set_text("Next")
    except Exception:
        pass

    status.set_text(f"Step {index+1} of {total}")


# guard to prevent re-entrancy
_running_next = {"busy": False}

async def on_next_click():
    if _running_next["busy"]:
        status.set_text("Already processing, please wait...")
        return
    _running_next["busy"] = True
    try:
        idx = _step["i"]
        section = CONFIG_JSON["sections"][idx]
        status.set_text(f"Processing '{section.get('title','')}' ... ⏳")

        # Resolve all fields in this section
        for fld in section.get("fields", []):
            fid = fld["id"]

            # Special case: URL step triggers scraping
            if fid == "business_site_url" and state_values.get("business_site_url"):
                await handle_scrape_and_set("business_site_url")

            # Process dependencies
            for dep in fld.get("depends_on", []):
                if dep == "scraped_text" and not state_values.get("scraped_text") and state_values.get("business_site_url"):
                    await handle_scrape_and_set("business_site_url")
                await resolve_dependents_for(dep)

            # Resolve the field itself
            await resolve_dependents_for(fid)

        # ✅ Check if all fields in this step have results
        all_ready = True
        for fld in section.get("fields", []):
            fid = fld["id"]
            if fld["type"] in ("text", "textarea"):
                if not state_values.get(fid, "").strip():
                    all_ready = False
            elif fld["type"] == "button_list":
                if not state_options.get(fid):
                    all_ready = False

        if all_ready:
            status.set_text(f"✅ Completed '{section.get('title','')}'")
            # advance step
            if idx < len(ordered_containers) - 1:
                show_step(idx + 1)
            else:
                status.set_text("🎉 Reached final step")
        else:
            status.set_text(f"⚠️ Cannot proceed: '{section.get('title','')}' not fully populated")

    except Exception as e:
        status.set_text(f"Next error: {e}")
    finally:
        _running_next["busy"] = False

# create one global Next button (place it where you want it on the page)
next_button = ui.button("Next", on_click=lambda: asyncio.create_task(on_next_click()))

# initialize: show only first step
show_step(0)



# Add Save JSON button
def save_state_to_file():
    payload = {
        "values": state_values,
        "options": state_options
    }
    fname = f"{state_values.get('business_name','business')}_dynamic_output.json".replace(" ", "_")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    status.set_text(f"Saved to {fname}")

ui.button("Save JSON", on_click=lambda: save_state_to_file())

# --------------------------
# Special handlers & dependency resolution
# --------------------------
async def handle_scrape_and_set(field_id: str):
    """
    If business_site_url changed, scrape homepage + a few internal pages,
    set 'scraped_text' and 'business_name' in state_values, then resolve dependents.
    """
    url = state_values.get(field_id, "").strip()
    if not url:
        status.set_text("No URL to scrape")
        return
    status.set_text("Scraping site (homepage + a few internal pages)... ⏳")
    try:
        r = session.get(url, timeout=10); r.raise_for_status()
        homepage_html = r.text
        soup = BeautifulSoup(homepage_html, "html.parser")
        title_tag = soup.title.string.strip() if soup.title and soup.title.string else ""
        business_name = title_tag.split("|")[0].strip() if title_tag else urlparse(url).netloc
        # build combined text
        combined = soup.get_text(separator="\n", strip=True)
        # grab relevant links and append up to 4 internal pages
        relevant = discover_relevant_links(url, homepage_html=homepage_html)
        for i, (k, link) in enumerate(list(relevant.items())[:4]):
            txt = scrape_text_from_url(link)
            if txt:
                combined += "\n\n" + txt
        # store scraped_text and business_name
        state_values["scraped_text"] = combined
        state_values["business_name"] = business_name
        status.set_text(f"Scraped site, extracted business_name='{business_name}'")
        # auto-fill a visible business_name widget if one exists
        if "business_name" in widgets:
            try:
                widgets["business_name"].value = business_name
            except Exception:
                pass
    except Exception as e:
        state_values["scraped_text"] = ""
        status.set_text(f"Scrape failed: {e}")
    # after scraping, resolve dependents of 'scraped_text' and of the URL itself
    await resolve_dependents_for("scraped_text")
    await resolve_dependents_for(field_id)

async def resolve_dependents_for(changed_field_id: str):
    """
    Resolve all fields that depend on changed_field_id.
    Each dependent will only be processed if all its dependencies are ready.
    Marks the section as complete once all its fields have values,
    and enables/disables the Next button accordingly.
    """
    dependents = dependency_map.get(changed_field_id, [])
    for dep_field_id in dependents:
        # --- locate field config
        target_cfg = None
        for sec in CONFIG_JSON["sections"]:
            for f in sec["fields"]:
                if f["id"] == dep_field_id:
                    target_cfg = f
                    break
            if target_cfg:
                break
        if not target_cfg:
            continue

        # --- check dependencies
        deps = target_cfg.get("depends_on", [])
        ready = True
        for d in deps:
            v = state_values.get(d, None)
            if v:
                continue
            if d in state_options and state_options[d]:
                continue
            if d == "scraped_text" and state_values.get("scraped_text", ""):
                continue
            ready = False
            break
        if not ready:
            continue

        # --- build prompt
        prompt_template = target_cfg.get("prompt", "")
        prompt_text = build_prompt_from_template(prompt_template)

        # --- call LLM
        target_type = target_cfg.get("type", "text")
        status.set_text(f"Running LLM for {dep_field_id} ... ⏳")
        result_text = await call_llm_text(prompt_text)
        status.set_text(f"LLM finished for {dep_field_id}")

        # --- handle text / textarea
        if target_type in ("text", "textarea"):
            # store raw trimmed result
            cleaned = (result_text or "").strip()
            state_values[dep_field_id] = cleaned

            # debug: show what LLM returned in server console
            print(f"[DEBUG] LLM result for {dep_field_id!r}: {repr(cleaned)[:800]}")

            widget = widgets.get(dep_field_id)
            if widget:
                try:
                    # correct update for NiceGUI input/textarea
                    widget.value = cleaned
                except Exception as e:
                    status.set_text(f"Update failed for {dep_field_id}: {e}")
                    print("Widget update error:", e)
            else:
                # fallback: create a visible label/markdown inside the corresponding section so user sees output
                print(f"[WARN] widget for {dep_field_id} not found in widgets dict. Rendering fallback output.")
                for sec in CONFIG_JSON["sections"]:
                    for f in sec["fields"]:
                        if f["id"] == dep_field_id:
                            try:
                                with section_containers[sec["title"]]:
                                    ui.markdown(f"**{dep_field_id} (generated):**\n\n{cleaned}")
                            except Exception:
                                pass
            # unhide the section this field belongs to
            for sec in CONFIG_JSON["sections"]:
                for f in sec["fields"]:
                    if f["id"] == dep_field_id:
                        try:
                            section_containers[sec["title"]].style("display:block")
                        except Exception:
                            pass

            # recurse to resolve deeper dependents
            await resolve_dependents_for(dep_field_id)


        # --- handle button_list
        elif target_type == "button_list":
            options = parse_list_response(result_text)
            state_options[dep_field_id] = options
            container = widgets.get(dep_field_id)
            if container:
                try:
                    container.clear()
                except Exception:
                    pass

                with container:
                    if not options:
                        ui.label("No options generated.")
                    else:
                        for opt in options:
                            def make_btn_click(fid=dep_field_id, o=opt):
                                async def _on_click():
                                    state_values[fid] = o
                                    status.set_text(f"Selected: {o}")
                                    await resolve_dependents_for(fid)
                                return _on_click
                            ui.button(opt, on_click=make_btn_click()).style(
                                "width:100%; margin-bottom:6px"
                            )

        # --- handle unknown type
        else:
            state_values[dep_field_id] = result_text.strip()
            w = widgets.get(dep_field_id)
            if w:
                try:
                    w.value = state_values[dep_field_id]
                except Exception:
                    pass
            await resolve_dependents_for(dep_field_id)

    # --- after processing dependents, check if the current section is complete
    for sec in CONFIG_JSON["sections"]:
        if any(f["id"] == changed_field_id for f in sec["fields"]):
            complete = True
            for fld in sec["fields"]:
                fid = fld["id"]
                if fld["type"] in ("text", "textarea"):
                    if not state_values.get(fid, "").strip():
                        complete = False
                elif fld["type"] == "button_list":
                    # Require that either options exist OR a value is chosen
                    if not state_options.get(fid) and not state_values.get(fid):
                        complete = False
            # ✅ Toggle Next button
            if complete:
                next_button.enable()
                status.set_text(f"✅ Section '{sec['title']}' is complete")
            else:
                next_button.disable()
            break

# Wire manual trigger: let user click a Run/Refresh button for the entire form (optional)
async def run_all_resolvers():
    # trigger resolution for all fields that have depends_on defined (attempt in order)
    for sec in CONFIG_JSON["sections"]:
        for f in sec["fields"]:
            for dep in f.get("depends_on", []):
                # if dep is scraped_text and scraped_text not present but business_site_url present, do scraping
                if dep == "scraped_text" and not state_values.get("scraped_text", "") and state_values.get("business_site_url", ""):
                    await handle_scrape_and_set("business_site_url")
                await resolve_dependents_for(dep)

ui.button("Run / Refresh All", on_click=lambda: asyncio.create_task(run_all_resolvers()))

# Set initial focus or instructions
status.set_text("Enter a Business Site URL and the form will auto-generate results.")

# --------------------------
# Start NiceGUI app
# --------------------------
if __name__ in {"__main__", "__mp_main__"}:
    print(">>> Starting UI now...")
    ui.run(title=CONFIG_JSON.get("title", "Dynamic App"), port=8080)
