#!/usr/bin/env python3
"""
main.py

Batch Q and A over PDFs in ./reports using Ollama vision models.

What this script does
  1. Loads prompts from ./LLM_query.json
  2. For each PDF:
       a. Starts a fresh chat memory
       b. Sends the initial instruction as a system message
       c. Sends each question as a separate user message, with the relevant PDF page image(s) attached
       d. Appends the assistant reply to the chat history so later questions have memory
       e. Saves results to JSON

Why /api/chat (not /api/generate)
  The /api/generate "context" parameter is deprecated and can be unreliable with multimodal inputs.
  /api/chat is the supported way to maintain conversational memory with images.

Folders
  ./reports   put PDFs here
  ./output    results written here
  ./static    created to avoid StaticFiles errors in other repos

Prompts file
  ./LLM_query.json

Ollama host
  Set OLLAMA_HOST to point at your Ollama server, for example:
    export OLLAMA_HOST=http://127.0.0.1:11434
  If not set, the script tries:
    http://127.0.0.1:11434
    http://localhost:11434
    http://[::1]:11434
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import requests
from tqdm import tqdm

fitz.TOOLS.mupdf_display_errors(False)
fitz.TOOLS.mupdf_display_warnings(False)

# =========================
# CONFIG
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = SCRIPT_DIR / "reports"
OUTPUT_DIR = SCRIPT_DIR / "output"
STATIC_DIR = SCRIPT_DIR / "static"

PROMPTS_PATH = Path(os.getenv("PROMPTS_PATH", str(SCRIPT_DIR / "LLM_query.json")))

MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:8b")

DPI = int(os.getenv("PDF_DPI", "200"))
IMG_FORMAT = os.getenv("IMG_FORMAT", "jpg")
JPG_QUALITY = int(os.getenv("JPG_QUALITY", "75"))

NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))

CONNECT_TIMEOUT_S = float(os.getenv("CONNECT_TIMEOUT_S", "10"))
READ_TIMEOUT_S = float(os.getenv("READ_TIMEOUT_S", "3600"))
REQUEST_TIMEOUT = (CONNECT_TIMEOUT_S, READ_TIMEOUT_S)

MAX_PAGES_FOR_Q1 = int(os.getenv("MAX_PAGES_FOR_Q1", "6"))
CONTINUE_ON_ERROR = os.getenv("CONTINUE_ON_ERROR", "1") == "1"

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
RETRY_BACKOFF_S = float(os.getenv("RETRY_BACKOFF_S", "2.0"))

CACHE_IMAGES = os.getenv("CACHE_IMAGES", "1") == "1"
AUTO_PULL_MODEL = os.getenv("AUTO_PULL_MODEL", "0") == "1"

LIMIT_PDFS = int(os.getenv("LIMIT_PDFS", "0"))  # 0 means no limit


# =========================
# Helpers
# =========================
def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)


def _normalise_base_url(value: str) -> str:
    v = value.strip()
    if not v:
        return v
    if v.startswith("http://") or v.startswith("https://"):
        return v.rstrip("/")
    return ("http://" + v).rstrip("/")


def resolve_ollama_base_url(session: requests.Session) -> str:
    env_host = os.getenv("OLLAMA_HOST", "").strip()
    candidates: List[str]
    if env_host:
        candidates = [_normalise_base_url(env_host)]
    else:
        candidates = [
            "http://127.0.0.1:11434",
            "http://localhost:11434",
            "http://[::1]:11434",
        ]

    last_err: Optional[str] = None
    for base in candidates:
        try:
            r = session.get(f"{base}/api/tags", timeout=CONNECT_TIMEOUT_S)
            if r.status_code == 200:
                return base
            last_err = f"HTTP {r.status_code}: {r.text}"
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"

    hint = (
        "Ollama is not reachable.\n\n"
        "Fix\n"
        "  1. Start the Ollama app\n"
        "  2. Confirm it is listening:\n"
        "     curl http://127.0.0.1:11434/api/tags\n"
        "  3. If Ollama runs elsewhere, set OLLAMA_HOST, for example:\n"
        "     export OLLAMA_HOST=http://192.168.1.50:11434\n\n"
        f"Tried: {', '.join(candidates)}\n"
    )
    if last_err:
        hint += f"\nLast error: {last_err}"
    raise RuntimeError(hint)


def pull_model(session: requests.Session, base_url: str, model: str) -> None:
    pull_url = f"{base_url}/api/pull"
    payload = {"model": model, "stream": True}

    tqdm.write(f"Pulling model via API: {model}")
    with session.post(pull_url, json=payload, stream=True, timeout=REQUEST_TIMEOUT) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Pull failed HTTP {r.status_code}: {r.text}".strip())
        last_status = None
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = obj.get("status")
            if status and status != last_status:
                tqdm.write(f"  {status}")
                last_status = status
            if status == "success":
                break


def ensure_model_available(session: requests.Session, base_url: str) -> None:
    tags_url = f"{base_url}/api/tags"

    def get_models() -> List[str]:
        payload = session.get(tags_url, timeout=CONNECT_TIMEOUT_S).json()
        models = [m.get("name") for m in payload.get("models", []) if isinstance(m, dict)]
        return [m for m in models if isinstance(m, str)]

    models = get_models()
    if MODEL in models:
        return

    tqdm.write(f"WARNING: Model '{MODEL}' not listed by /api/tags")

    if AUTO_PULL_MODEL:
        pull_model(session, base_url, MODEL)
        models = get_models()
        if MODEL in models:
            return

    raise RuntimeError(
        f"Model '{MODEL}' is not available in Ollama.\n\n"
        "Fix\n"
        f"  curl {base_url}/api/pull -d '{{\"model\":\"{MODEL}\"}}'\n"
        "or set a different model:\n"
        "  export OLLAMA_MODEL=<installed model>\n\n"
        "Installed models:\n  " + "\n  ".join(models[:80] if models else ["(none)"])
    )


def render_pdf_to_images(pdf_path: Path) -> List[Path]:
    out_dir = OUTPUT_DIR / "cache_images" / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    page_paths: List[Path] = []

    render_bar = tqdm(
        range(doc.page_count),
        desc=f"Render {pdf_path.name}",
        unit="pg",
        leave=False,
        dynamic_ncols=True,
    )
    for i in render_bar:
        img_path = out_dir / f"page_{i+1:03d}.{IMG_FORMAT}"
        if CACHE_IMAGES and img_path.exists():
            page_paths.append(img_path)
            continue

        page = doc.load_page(i)
        mat = fitz.Matrix(DPI / 72.0, DPI / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        if IMG_FORMAT.lower() in {"jpg", "jpeg"}:
            pix.save(str(img_path), output="jpeg", jpg_quality=JPG_QUALITY)
        else:
            pix.save(str(img_path))

        page_paths.append(img_path)

    doc.close()
    return page_paths


def images_to_b64(image_paths: List[Path]) -> List[str]:
    return [base64.b64encode(p.read_bytes()).decode("utf-8") for p in image_paths]


def select_image_paths(page_imgs: List[Path], page_sel: Any, step_id: str) -> List[Path]:
    if page_sel == "all":
        if str(step_id).endswith("Q1") and MAX_PAGES_FOR_Q1 > 0:
            return page_imgs[:MAX_PAGES_FOR_Q1]
        return page_imgs

    if isinstance(page_sel, int):
        idx = page_sel - 1
        if 0 <= idx < len(page_imgs):
            return [page_imgs[idx]]
        return page_imgs

    return page_imgs


# =========================
# Prompt loading
# =========================
def _phase_to_page_sel(phase: Optional[str]) -> Optional[Any]:
    if not phase:
        return None
    if phase == "full_pdf":
        return "all"
    m = re.match(r"^page(\d+)$", phase)
    if m:
        return int(m.group(1))
    return None


def normalise_prompt_plan(raw: Dict[str, Any]) -> Dict[str, Any]:
    # Plan format
    if isinstance(raw, dict) and "instruction_prompt" in raw and "steps" in raw:
        steps_out: List[Dict[str, Any]] = []
        for i, s in enumerate(raw.get("steps") or []):
            if isinstance(s, dict):
                steps_out.append(
                    {
                        "step_id": s.get("step_id") or f"STEP_{i+1:03d}",
                        "kind": s.get("kind") or "question",
                        "prompt": (s.get("prompt") or "").strip(),
                        "page_sel": s.get("page_sel"),
                        "use_images": bool(s.get("use_images", True)),
                        "phase": s.get("phase"),
                    }
                )
        return {
            "meta": raw.get("meta", {}),
            "instruction_prompt": (raw.get("instruction_prompt") or "").strip(),
            "steps": steps_out,
        }

    # Flat prompts list
    prompts = raw.get("prompts")
    if isinstance(prompts, list):
        instruction_prompt = ""
        steps_out: List[Dict[str, Any]] = []

        for i, item in enumerate(prompts):
            if not isinstance(item, dict):
                continue

            pid = item.get("id") or item.get("qnum") or f"STEP_{i+1:03d}"
            phase = item.get("phase")
            kind = item.get("type") or item.get("kind") or "question"
            content = (item.get("content") or item.get("prompt") or "").strip()

            if kind == "instruction" and (phase == "system" or not instruction_prompt):
                if not instruction_prompt and content:
                    instruction_prompt = content
                    continue

            page_sel = item.get("page_sel")
            if page_sel is None:
                page_sel = _phase_to_page_sel(phase)

            use_images = bool(item.get("use_images", True))
            if kind == "instruction":
                use_images = False

            steps_out.append(
                {
                    "step_id": str(pid),
                    "kind": "instruction" if kind == "instruction" else "question",
                    "prompt": content,
                    "page_sel": page_sel,
                    "use_images": use_images,
                    "phase": phase,
                }
            )

        if not instruction_prompt:
            instruction_prompt = (raw.get("system_prompt") or "").strip()

        if not instruction_prompt:
            raise ValueError("No instruction prompt found in LLM_query.json.")

        return {"meta": raw.get("meta", {}), "instruction_prompt": instruction_prompt, "steps": steps_out}

    # Prompts only by phases format
    if "system_prompt" in raw and "phases" in raw and isinstance(raw["phases"], dict):
        instruction_prompt = (raw.get("system_prompt") or "").strip()
        steps_out: List[Dict[str, Any]] = []
        for phase, pobj in raw["phases"].items():
            if not isinstance(pobj, dict):
                continue
            for qk, qprompt in (pobj.get("prompts") or {}).items():
                steps_out.append(
                    {
                        "step_id": f"{phase}_{qk}",
                        "kind": "question",
                        "prompt": (qprompt or "").strip(),
                        "page_sel": _phase_to_page_sel(phase),
                        "use_images": True,
                        "phase": phase,
                    }
                )
        if not instruction_prompt:
            raise ValueError("No system_prompt found in LLM_query.json.")
        return {"meta": raw.get("meta", {}), "instruction_prompt": instruction_prompt, "steps": steps_out}

    raise ValueError("Unsupported LLM_query.json format.")


def load_prompt_plan() -> Dict[str, Any]:
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing prompts file: {PROMPTS_PATH}\n"
            "Place your prompt JSON there or set PROMPTS_PATH to the correct path."
        )
    raw = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    return normalise_prompt_plan(raw)


def copy_prompts_used(plan: Dict[str, Any]) -> None:
    (OUTPUT_DIR / "prompts_used.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =========================
# Ollama chat (streaming)
# =========================
@dataclass
class OllamaOptions:
    num_ctx: int = NUM_CTX
    temperature: float = TEMPERATURE


def ollama_chat_stream(
    session: requests.Session,
    chat_url: str,
    messages: List[Dict[str, Any]],
    options: OllamaOptions,
    progress: Optional[tqdm],
    step_label: str,
) -> str:
    payload: Dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": "5m",
        "options": {
            "temperature": options.temperature,
            "num_ctx": options.num_ctx,
        },
    }

    with session.post(chat_url, json=payload, stream=True, timeout=REQUEST_TIMEOUT) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text}".strip())

        parts: List[str] = []
        char_count = 0

        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            if obj.get("error"):
                raise RuntimeError(f"Ollama error: {obj['error']}")

            msg = obj.get("message") or {}
            chunk = msg.get("content") or ""
            if chunk:
                parts.append(chunk)
                char_count += len(chunk)
                if progress is not None:
                    progress.set_postfix(step=step_label, out=f"{char_count}ch")

            if obj.get("done"):
                break

        return "".join(parts).strip()


def chat_with_retries(
    session: requests.Session,
    chat_url: str,
    messages: List[Dict[str, Any]],
    step_bar: Optional[tqdm],
    step_id: str,
) -> str:
    opts = OllamaOptions()

    for attempt in range(MAX_RETRIES + 1):
        try:
            if step_bar is not None:
                step_bar.set_postfix(step=step_id, try_=attempt + 1)
            return ollama_chat_stream(
                session=session,
                chat_url=chat_url,
                messages=messages,
                options=opts,
                progress=step_bar,
                step_label=step_id,
            )
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
            err = f"{type(e).__name__}: {e}"
        except requests.RequestException as e:
            err = f"{type(e).__name__}: {e}"
        except RuntimeError as e:
            err = str(e)

        if attempt >= MAX_RETRIES:
            raise RuntimeError(f"{step_id} failed after {MAX_RETRIES + 1} attempts: {err}")

        # Light backoff
        time.sleep(RETRY_BACKOFF_S * (2 ** attempt))

    raise RuntimeError(f"{step_id} failed unexpectedly.")


# =========================
# Main
# =========================
def main() -> None:
    ensure_dirs()

    plan = load_prompt_plan()
    copy_prompts_used(plan)

    instruction_prompt: str = (plan.get("instruction_prompt") or "").strip()
    steps: List[Dict[str, Any]] = list(plan.get("steps") or [])

    pdfs = sorted(REPORTS_DIR.glob("*.pdf"))
    if LIMIT_PDFS > 0:
        pdfs = pdfs[:LIMIT_PDFS]

    if not pdfs:
        print(f"No PDFs found in: {REPORTS_DIR}")
        return

    session = requests.Session()
    base_url = resolve_ollama_base_url(session)
    ensure_model_available(session, base_url)

    chat_url = f"{base_url}/api/chat"

    agg_jsonl_path = OUTPUT_DIR / "all_answers.jsonl"
    agg_json_path = OUTPUT_DIR / "all_answers.json"
    all_outputs: List[Dict[str, Any]] = []

    with open(agg_jsonl_path, "w", encoding="utf-8") as jf:
        for pdf in tqdm(pdfs, desc="PDFs", unit="pdf", dynamic_ncols=True):
            page_img_paths = render_pdf_to_images(pdf)

            results: List[Dict[str, Any]] = []

            # Fresh memory per PDF
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": instruction_prompt}
            ]

            # Record the system instruction
            results.append(
                {
                    "step_id": "INIT",
                    "kind": "instruction",
                    "phase": "system",
                    "page_sel": None,
                    "pages_sent": [],
                    "prompt": instruction_prompt,
                    "sent_message_preview": instruction_prompt[:300],
                    "response": "",
                    "seconds": 0.0,
                    "model": MODEL,
                    "ollama_base_url": base_url,
                }
            )

            step_bar = tqdm(
                steps,
                desc=f"Prompts: {pdf.name}",
                unit="p",
                leave=False,
                dynamic_ncols=True,
            )

            for step in step_bar:
                step_id = str(step.get("step_id", ""))
                kind = step.get("kind", "question")
                phase = step.get("phase")
                page_sel = step.get("page_sel", None)
                prompt = (step.get("prompt") or "").strip()
                use_images = bool(step.get("use_images", True))

                chosen_pages: List[int] = []
                imgs_b64: List[str] = []

                if use_images and kind != "instruction":
                    chosen_paths = select_image_paths(page_img_paths, page_sel, step_id)
                    imgs_b64 = images_to_b64(chosen_paths)
                    chosen_pages = [
                        int(p.stem.split("_")[-1])
                        for p in chosen_paths
                        if p.stem.startswith("page_") and p.stem.split("_")[-1].isdigit()
                    ]

                user_message: Dict[str, Any] = {"role": "user", "content": prompt}
                if imgs_b64:
                    user_message["images"] = imgs_b64

                # Append message, call, then append assistant reply
                messages.append(user_message)

                t0 = perf_counter()
                try:
                    ans = chat_with_retries(
                        session=session,
                        chat_url=chat_url,
                        messages=messages,
                        step_bar=step_bar,
                        step_id=step_id,
                    )
                    dt = perf_counter() - t0

                    # Add assistant response into memory
                    messages.append({"role": "assistant", "content": ans})

                    results.append(
                        {
                            "step_id": step_id,
                            "kind": kind,
                            "phase": phase,
                            "page_sel": page_sel,
                            "pages_sent": chosen_pages,
                            "prompt": prompt,
                            "sent_message_preview": (prompt[:300] if prompt else ""),
                            "response": ans,
                            "seconds": round(dt, 3),
                        }
                    )
                except Exception as e:
                    dt = perf_counter() - t0
                    err_text = f"ERROR: {type(e).__name__}: {e}"

                    # Keep memory consistent: remove the last user message if it failed
                    if messages and messages[-1] == user_message:
                        messages.pop()

                    results.append(
                        {
                            "step_id": step_id,
                            "kind": kind,
                            "phase": phase,
                            "page_sel": page_sel,
                            "pages_sent": chosen_pages,
                            "prompt": prompt,
                            "sent_message_preview": (prompt[:300] if prompt else ""),
                            "response": err_text,
                            "seconds": round(dt, 3),
                        }
                    )

                    tqdm.write(f"[{pdf.name}] {step_id} -> {err_text}")

                    if not CONTINUE_ON_ERROR:
                        raise

            record = {"file": pdf.name, "results": results}

            (OUTPUT_DIR / f"{pdf.stem}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (OUTPUT_DIR / f"{pdf.stem}.txt").write_text(
                "\n".join(f"{r['step_id']}: {r['response']}" for r in results) + "\n",
                encoding="utf-8",
            )

            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            all_outputs.append(record)

            tqdm.write(f"Processed: {pdf.name}")

    agg_json_path.write_text(
        json.dumps(all_outputs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Ollama base URL: {base_url}")
    print(f"Reports folder:  {REPORTS_DIR}")
    print(f"Output folder:   {OUTPUT_DIR}")
    print(f"Aggregate JSONL: {agg_jsonl_path}")
    print(f"Aggregate JSON:  {agg_json_path}")
    print(f"Prompts used:    {OUTPUT_DIR / 'prompts_used.json'}")
    if LIMIT_PDFS > 0:
        print(f"LIMIT_PDFS was set, processed: {LIMIT_PDFS}")


if __name__ == "__main__":
    main()
