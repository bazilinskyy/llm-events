#!/usr/bin/env python3
"""
main.py

Batch Q&A over traffic accident report PDFs using a local Ollama VISION model.

What this version fixes:
- Adds tqdm progress bars (PDFs, page rendering, per-question steps)
- Uses streaming responses from Ollama so you can see activity (and avoids “silent” hangs)
- Uses smaller JPG images by default (much faster + less likely to trigger 500/timeouts)
- Adds retries + graceful error capture so one bad call doesn’t kill the whole run
- Caps how many pages are sent for Q1 (configurable) to avoid huge multimodal payloads

Folders:
  ./reports   -> put PDFs here
  ./output    -> results written here
  ./static    -> created to avoid StaticFiles errors in other repos

Prompt spec:
  Put the full prompt text into: ./prompt_spec.txt
  (This keeps the script readable and avoids giant code blocks.)
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

# Silence MuPDF stderr noise about malformed PDFs
fitz.TOOLS.mupdf_display_errors(False)
fitz.TOOLS.mupdf_display_warnings(False)


# =========================
# CONFIG
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = SCRIPT_DIR / "reports"
OUTPUT_DIR = SCRIPT_DIR / "output"
STATIC_DIR = SCRIPT_DIR / "static"

PROMPT_SPEC_PATH = SCRIPT_DIR / "prompt_spec.txt"

# Ollama endpoint (match your serve log: 127.0.0.1)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_GENERATE_URL = f"{OLLAMA_HOST}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"

# Model (must support images)
MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:8b")

# Image rendering (smaller => faster, fewer 500s/timeouts)
DPI = int(os.getenv("PDF_DPI", "200"))         # try 150–200 on laptops
IMG_FORMAT = os.getenv("IMG_FORMAT", "jpg")    # "jpg" recommended
JPG_QUALITY = int(os.getenv("JPG_QUALITY", "75"))

# IMPORTANT: your Ollama log shows default_num_ctx=4096 on your GPU
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

# Network timeouts
CONNECT_TIMEOUT_S = float(os.getenv("CONNECT_TIMEOUT_S", "10"))
READ_TIMEOUT_S = float(os.getenv("READ_TIMEOUT_S", "3600"))  # 1 hour
REQUEST_TIMEOUT = (CONNECT_TIMEOUT_S, READ_TIMEOUT_S)

# Reduce payload size for Q1 (set 0 to send all pages)
MAX_PAGES_FOR_Q1 = int(os.getenv("MAX_PAGES_FOR_Q1", "6"))

# If True, keep going even if a question fails (records ERROR in output)
CONTINUE_ON_ERROR = os.getenv("CONTINUE_ON_ERROR", "1") == "1"

# Retries on 500/timeouts
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
RETRY_BACKOFF_S = float(os.getenv("RETRY_BACKOFF_S", "2.0"))

# Cache rendered images to disk so reruns are faster
CACHE_IMAGES = os.getenv("CACHE_IMAGES", "1") == "1"


# ============================================================
# Prompt spec loader
# ============================================================
def load_prompt_spec() -> str:
    if not PROMPT_SPEC_PATH.exists():
        raise FileNotFoundError(
            f"Missing prompt spec file: {PROMPT_SPEC_PATH}\n"
            "Create it and paste your full PROMPT_PDF_TEXT into it (exactly as before)."
        )
    return PROMPT_SPEC_PATH.read_text(encoding="utf-8")


# =========================
# Helpers
# =========================
def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)


def check_ollama(session: requests.Session) -> None:
    """Fail fast if Ollama is not reachable."""
    try:
        r = session.get(OLLAMA_TAGS_URL, timeout=CONNECT_TIMEOUT_S)
    except requests.RequestException as e:
        raise RuntimeError(
            f"Ollama is not reachable at {OLLAMA_HOST}. Start it with: ollama serve"
        ) from e

    if r.status_code != 200:
        raise RuntimeError(f"Ollama responded with HTTP {r.status_code}: {r.text}")

    # Optional: warn if model not found
    try:
        data = r.json()
        models = [m.get("name") for m in data.get("models", []) if isinstance(m, dict)]
        if MODEL not in models:
            tqdm.write(
                f"WARNING: Model '{MODEL}' not listed by /api/tags. "
                f"If requests fail, run: ollama pull {MODEL}"
            )
    except Exception:
        pass


def render_pdf_to_images(pdf_path: Path) -> List[Path]:
    """
    Render each page to an image file (cached).
    Returns list of image paths in page order.
    """
    img_dir = OUTPUT_DIR / "_images" / pdf_path.stem
    img_dir.mkdir(parents=True, exist_ok=True)

    # Determine extension
    fmt = IMG_FORMAT.lower()
    if fmt not in {"png", "jpg", "jpeg"}:
        raise ValueError("IMG_FORMAT must be png or jpg/jpeg")

    ext = "jpg" if fmt in {"jpg", "jpeg"} else "png"

    doc = fitz.open(str(pdf_path))
    try:
        total_pages = doc.page_count
        scale = DPI / 72.0
        mat = fitz.Matrix(scale, scale)

        out_paths: List[Path] = []
        for i in tqdm(
            range(total_pages),
            desc=f"Render pages: {pdf_path.name}",
            unit="page",
            leave=False,
            dynamic_ncols=True,
        ):
            out_file = img_dir / f"page_{i+1:03d}.{ext}"
            out_paths.append(out_file)

            if CACHE_IMAGES and out_file.exists():
                continue

            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            if ext == "jpg":
                img_bytes = pix.tobytes("jpg", JPG_QUALITY)
            else:
                img_bytes = pix.tobytes("png")

            out_file.write_bytes(img_bytes)

        return out_paths
    finally:
        doc.close()


def images_to_b64(image_paths: List[Path]) -> List[str]:
    out: List[str] = []
    for p in image_paths:
        out.append(base64.b64encode(p.read_bytes()).decode("utf-8"))
    return out


# ============================================================
# Prompt spec parsing (kept from your script)
# ============================================================
def extract_sections(spec: str) -> Dict[str, str]:
    m1 = spec.find("page1")
    m2 = spec.find("page2")
    m3 = spec.find("page3")

    if m1 == -1 or m2 == -1 or m3 == -1:
        return {"global": spec, "page1": "", "page2": "", "page3": ""}

    b1 = spec.rfind("\n", 0, m1)
    b2 = spec.rfind("\n", 0, m2)
    b3 = spec.rfind("\n", 0, m3)
    b1 = 0 if b1 == -1 else b1 + 1
    b2 = 0 if b2 == -1 else b2 + 1
    b3 = 0 if b3 == -1 else b3 + 1

    return {
        "global": spec[:b1],
        "page1": spec[b1:b2],
        "page2": spec[b2:b3],
        "page3": spec[b3:],
    }


def extract_q_blocks(text: str) -> List[str]:
    matches = list(re.finditer(r"(?m)^Q\d+\.", text))
    if not matches:
        return []
    blocks: List[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append(text[start:end].strip("\n"))
    return blocks


def base_rules_text(global_section: str) -> str:
    key = "============================First, give the entire"
    idx = global_section.find(key)
    if idx == -1:
        qidx = global_section.find("Q1.")
        return global_section[:qidx].strip("\n") if qidx != -1 else global_section.strip("\n")
    return global_section[:idx].strip("\n")


def page_header_text(section: str) -> str:
    qidx = section.find("Q")
    return section[:qidx].strip("\n") if qidx != -1 else section.strip("\n")


def parse_v2_id(q9_answer: str) -> Optional[str]:
    m = re.search(r"\bv2_id\s*=\s*([A-Za-z_]+)", q9_answer)
    return m.group(1).strip().lower() if m else None


def build_steps(spec: str) -> List[Tuple[str, Any, str]]:
    sections = extract_sections(spec)

    g = sections["global"]
    p1 = sections["page1"]
    p2 = sections["page2"]
    p3 = sections["page3"]

    rules = base_rules_text(g)
    q_global = extract_q_blocks(g)
    q1_block = q_global[0] if q_global else ""

    p1_hdr = page_header_text(p1)
    p2_hdr = page_header_text(p2)
    p3_hdr = page_header_text(p3)

    b1 = extract_q_blocks(p1)  # expected 8
    b2 = extract_q_blocks(p2)  # expected 9
    b3 = extract_q_blocks(p3)  # expected 7

    if len(b1) != 8 or len(b2) != 9 or len(b3) != 7 or not q1_block:
        tqdm.write("WARNING: Prompt spec parsing counts unexpected.")
        tqdm.write(f"  global Q blocks: {len(q_global)} (need 1)")
        tqdm.write(f"  page1 Q blocks:  {len(b1)} (need 8)")
        tqdm.write(f"  page2 Q blocks:  {len(b2)} (need 9)")
        tqdm.write(f"  page3 Q blocks:  {len(b3)} (need 7)")

    steps: List[Tuple[str, Any, str]] = []

    # Q1: entire report
    steps.append(("Q1", "all", (rules + "\n\n" + q1_block).strip("\n")))

    # Page 1: Q2..Q8 (note: b1[0] is labeled Q1 in the spec but is vehicle-info prompt)
    if len(b1) >= 8:
        steps.append(("Q2_vehicle", 1, (rules + "\n\n" + p1_hdr + "\n\n" + b1[0]).strip("\n")))
        steps.append(("Q2_time",    1, (rules + "\n\n" + p1_hdr + "\n\n" + b1[1]).strip("\n")))
        steps.append(("Q3",         1, (rules + "\n\n" + p1_hdr + "\n\n" + b1[2]).strip("\n")))

        # Q4-Q6 require online search; local-only => answer NA without calling model
        steps.append(("Q4", "local_na", b1[3]))
        steps.append(("Q5", "local_na", b1[4]))
        steps.append(("Q6", "local_na", b1[5]))

        steps.append(("Q7",         1, (rules + "\n\n" + p1_hdr + "\n\n" + b1[6]).strip("\n")))
        steps.append(("Q8",         1, (rules + "\n\n" + p1_hdr + "\n\n" + b1[7]).strip("\n")))

    # Page 2: Q9..Q17
    if len(b2) >= 9:
        steps.append(("Q9",  2, (rules + "\n\n" + p2_hdr + "\n\n" + b2[0]).strip("\n")))
        steps.append(("Q10", 2, (rules + "\n\n" + p2_hdr + "\n\n" + b2[1]).strip("\n")))
        steps.append(("Q11", 2, (rules + "\n\n" + p2_hdr + "\n\n" + b2[2]).strip("\n")))
        steps.append(("Q12", 2, (rules + "\n\n" + p2_hdr + "\n\n" + b2[3]).strip("\n")))
        steps.append(("Q13", 2, (rules + "\n\n" + p2_hdr + "\n\n" + b2[4]).strip("\n")))
        steps.append(("Q14", 2, (rules + "\n\n" + p2_hdr + "\n\n" + b2[5]).strip("\n")))
        steps.append(("Q15", 2, (rules + "\n\n" + p2_hdr + "\n\n" + b2[6]).strip("\n")))
        steps.append(("Q16", 2, (rules + "\n\n" + p2_hdr + "\n\n" + b2[7]).strip("\n")))
        steps.append(("Q17", 2, (rules + "\n\n" + p2_hdr + "\n\n" + b2[8]).strip("\n")))

    # Page 3: Q18..Q24
    if len(b3) >= 7:
        steps.append(("Q18", 3, (rules + "\n\n" + p3_hdr + "\n\n" + b3[0]).strip("\n")))
        steps.append(("Q19", 3, (rules + "\n\n" + p3_hdr + "\n\n" + b3[1]).strip("\n")))
        steps.append(("Q20", 3, (rules + "\n\n" + p3_hdr + "\n\n" + b3[2]).strip("\n")))
        steps.append(("Q21", 3, (rules + "\n\n" + p3_hdr + "\n\n" + b3[3]).strip("\n")))
        steps.append(("Q22", 3, (rules + "\n\n" + p3_hdr + "\n\n" + b3[4]).strip("\n")))
        steps.append(("Q23", 3, (rules + "\n\n" + p3_hdr + "\n\n" + b3[5]).strip("\n")))
        steps.append(("Q24", 3, (rules + "\n\n" + p3_hdr + "\n\n" + b3[6]).strip("\n")))

    return steps


def select_image_paths(page_imgs: List[Path], page_sel: Any, step_id: str) -> List[Path]:
    """
    page_sel:
      - "all" => all pages (optionally capped for Q1)
      - int N => page N
      - "local_na" => handled outside
    """
    if page_sel == "all":
        if step_id == "Q1" and MAX_PAGES_FOR_Q1 > 0:
            return page_imgs[:MAX_PAGES_FOR_Q1]
        return page_imgs

    if isinstance(page_sel, int):
        idx = page_sel - 1
        if 0 <= idx < len(page_imgs):
            return [page_imgs[idx]]
        return page_imgs

    return page_imgs


# ============================================================
# Ollama streaming + retries
# ============================================================
@dataclass
class OllamaOptions:
    num_ctx: int = NUM_CTX
    temperature: float = 0.0


def ollama_generate_stream(
    session: requests.Session,
    prompt: str,
    images_b64: List[str],
    options: OllamaOptions,
    progress: Optional[tqdm] = None,
    step_label: str = "",
) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": images_b64,
        "stream": True,          # streaming JSON lines
        "keep_alive": "5m",
        "options": {
            "temperature": options.temperature,
            "num_ctx": options.num_ctx,
        },
    }

    with session.post(
        OLLAMA_GENERATE_URL,
        json=payload,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as r:
        if r.status_code != 200:
            # Try to include response text (may contain the reason)
            txt = ""
            try:
                txt = r.text
            except Exception:
                pass
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {txt}".strip())

        parts: List[str] = []
        char_count = 0
        saw_any = False

        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if obj.get("error"):
                raise RuntimeError(f"Ollama error: {obj['error']}")

            chunk = obj.get("response") or ""
            if chunk:
                saw_any = True
                parts.append(chunk)
                char_count += len(chunk)
                if progress is not None:
                    progress.set_postfix(step=step_label, out=f"{char_count}ch")

            if obj.get("done"):
                break

        # If it ended without emitting, still return whatever we got (possibly empty)
        if progress is not None and not saw_any:
            progress.set_postfix(step=step_label, out="0ch")

        return "".join(parts).strip()


def generate_with_retries(
    session: requests.Session,
    prompt: str,
    images_b64: List[str],
    step_bar: Optional[tqdm],
    step_id: str,
) -> str:
    """
    Retries on timeouts/500s and applies fallbacks:
      - reduce num_ctx (never above 4096 by default)
      - reduce number of images (especially for Q1)
    """
    base_opts = OllamaOptions(num_ctx=NUM_CTX, temperature=0.0)
    current_images = images_b64

    for attempt in range(MAX_RETRIES + 1):
        try:
            if step_bar is not None:
                step_bar.set_postfix(step=step_id, try_=attempt + 1)
            return ollama_generate_stream(
                session=session,
                prompt=prompt,
                images_b64=current_images,
                options=base_opts,
                progress=step_bar,
                step_label=step_id,
            )
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
            err = f"{type(e).__name__}: {e}"
        except requests.RequestException as e:
            err = f"{type(e).__name__}: {e}"
        except RuntimeError as e:
            err = str(e)

        # Last attempt => raise
        if attempt >= MAX_RETRIES:
            raise RuntimeError(f"{step_id} failed after {MAX_RETRIES+1} attempts: {err}")

        # Fallbacks for next try
        # 1) Keep num_ctx reasonable (your server default was 4096)
        base_opts.num_ctx = min(base_opts.num_ctx, 4096)

        # 2) If lots of images, reduce for the retry (common cause of 500)
        if len(current_images) > 3:
            current_images = current_images[:3]

        # backoff
        time.sleep(RETRY_BACKOFF_S * (2**attempt))

    raise RuntimeError(f"{step_id} failed unexpectedly.")


# ============================================================
# Main
# ============================================================
def main() -> None:
    ensure_dirs()

    spec = load_prompt_spec()
    steps = build_steps(spec)

    pdfs = sorted(REPORTS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in: {REPORTS_DIR}")
        print("Put your report PDFs into that folder and run again.")
        return

    session = requests.Session()
    check_ollama(session)

    agg_path = OUTPUT_DIR / "all_answers.jsonl"

    with open(agg_path, "w", encoding="utf-8") as jf:
        for pdf in tqdm(pdfs, desc="PDFs", unit="pdf", dynamic_ncols=True):
            # Render pages (cached) with progress
            page_img_paths = render_pdf_to_images(pdf)

            results: List[Dict[str, Any]] = []
            q9_answer_text = ""

            step_bar = tqdm(
                steps,
                desc=f"Questions: {pdf.name}",
                unit="q",
                leave=False,
                dynamic_ncols=True,
            )

            for step_id, page_sel, prompt in step_bar:
                step_bar.set_postfix(step=step_id)

                # Local-only: online-search questions are answered as NA
                if page_sel == "local_na":
                    if step_id == "Q4":
                        ans = "Q4. Lane number=NA, Street type=NA."
                    elif step_id == "Q5":
                        ans = "Q5. Speed=NA."
                    elif step_id == "Q6":
                        ans = "Q6. street_busy=NA."
                    else:
                        ans = f"{step_id}=NA"
                    results.append({"step_id": step_id, "response": ans, "seconds": 0.0})
                    continue

                # Skip Q10 if Q9 indicates v2_id=vehicle
                if step_id == "Q10":
                    v2_id = parse_v2_id(q9_answer_text)
                    if v2_id == "vehicle":
                        results.append({"step_id": "Q10", "response": "SKIPPED (v2_id==vehicle)", "seconds": 0.0})
                        continue

                # Select images for this step (cap Q1 pages if configured)
                chosen_paths = select_image_paths(page_img_paths, page_sel, step_id)
                imgs_b64 = images_to_b64(chosen_paths)

                t0 = perf_counter()
                try:
                    ans = generate_with_retries(
                        session=session,
                        prompt=prompt,
                        images_b64=imgs_b64,
                        step_bar=step_bar,
                        step_id=step_id,
                    )
                    dt = perf_counter() - t0
                    results.append({"step_id": step_id, "response": ans, "seconds": round(dt, 3)})

                    if step_id == "Q9":
                        q9_answer_text = ans

                except Exception as e:
                    dt = perf_counter() - t0
                    err_text = f"ERROR: {type(e).__name__}: {e}"
                    results.append({"step_id": step_id, "response": err_text, "seconds": round(dt, 3)})

                    tqdm.write(f"[{pdf.name}] {step_id} -> {err_text}")

                    if not CONTINUE_ON_ERROR:
                        raise

            # Save per-PDF outputs
            (OUTPUT_DIR / f"{pdf.stem}.json").write_text(
                json.dumps({"file": pdf.name, "results": results}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (OUTPUT_DIR / f"{pdf.stem}.txt").write_text(
                "\n".join(f"{r['step_id']}: {r['response']}" for r in results) + "\n",
                encoding="utf-8",
            )

            jf.write(json.dumps({"file": pdf.name, "results": results}, ensure_ascii=False) + "\n")
            tqdm.write(f"Processed: {pdf.name}")

    print("\nDone.")
    print(f"Reports folder:  {REPORTS_DIR}")
    print(f"Output folder:   {OUTPUT_DIR}")
    print(f"Aggregate JSONL: {agg_path}")
    print("\nTip: If Q1 is still slow or causes 500s, reduce MAX_PAGES_FOR_Q1 (env var) or DPI.")


if __name__ == "__main__":
    main()