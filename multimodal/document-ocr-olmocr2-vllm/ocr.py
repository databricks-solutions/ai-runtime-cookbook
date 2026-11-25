"""PDF -> Markdown at scale with olmOCR 2 (a Qwen2.5-VL fine-tune for documents) and vLLM.

Renders each PDF page to an image, asks olmOCR 2 for Markdown (tables as HTML, equations as
LaTeX) with a YAML front matter of page metadata, and writes per-page Markdown files plus a
Parquet table to a Unity Catalog volume.
"""

import base64
import io
import json
import os
import time
from itertools import zip_longest
from pathlib import Path

import mlflow
import pandas as pd
import pypdfium2 as pdfium
import yaml
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

# The olmOCR v4 page prompt, from olmocr.prompts.build_no_anchoring_v4_yaml_prompt (Apache-2.0).
PROMPT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Convert equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax "
    "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top specifying values for the "
    "primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
)


def render_page(pdf, index):
    """Render a page so its longest side is target_longest_image_dim pixels, as olmOCR expects."""
    page = pdf[index]
    width, height = page.get_size()
    scale = P["target_longest_image_dim"] / max(width, height)
    image = page.render(scale=scale).to_pil()
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def split_front_matter(text):
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            try:
                return yaml.safe_load(parts[1]) or {}, parts[2].strip()
            except yaml.YAMLError:
                pass
    return {}, text.strip()


def main():
    root = snapshot_download(
        "allenai/olmOCR-bench", repo_type="dataset",
        allow_patterns=[f"bench_data/pdfs/{c}/*.pdf" for c in P["pdf_categories"]],
    )
    # Take PDFs from each category in turn so every category is in the sample.
    by_category = [sorted((Path(root) / "bench_data/pdfs" / c).glob("*.pdf")) for c in P["pdf_categories"]]
    # Drops the None padding zip_longest adds when the categories have different sizes.
    pdfs = [pdf for group in zip_longest(*by_category) for pdf in group if pdf][: P["max_pdfs"]]
    print(f"[sample] {len(pdfs)} PDFs: " + ", ".join(
        f"{c} {sum(p.parent.name == c for p in pdfs)}" for c in P["pdf_categories"]))

    t0 = time.time()
    pages, requests = [], []
    for path in pdfs:
        pdf = pdfium.PdfDocument(path)
        for i in range(len(pdf)):
            pages.append({"doc_id": f"{path.parent.name}/{path.stem}", "page": i + 1})
            requests.append([{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{render_page(pdf, i)}"}},
            ]}])
        pdf.close()
    render_s = time.time() - t0
    print(f"[render] {len(pages)} pages from {len(pdfs)} PDFs in {render_s:.0f}s")

    llm = LLM(model=P["model_name"], max_model_len=16384, limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=0.85)
    t0 = time.time()
    outs = llm.chat(requests, SamplingParams(temperature=0.1, max_tokens=P["max_tokens"]))
    infer_s = time.time() - t0

    out = Path(P["output_root"]) / os.environ["MLFLOW_RUN_ID"]
    md_dir = out / "markdown"
    rows, finished = [], 0
    for meta, o in zip(pages, outs):
        front, markdown = split_front_matter(o.outputs[0].text)
        finished += o.outputs[0].finish_reason == "stop"
        rows.append({**meta, "markdown": markdown, "primary_language": front.get("primary_language"),
                     "is_table": front.get("is_table"), "is_diagram": front.get("is_diagram"),
                     "rotation_correction": front.get("rotation_correction"),
                     "output_tokens": len(o.outputs[0].token_ids)})
        dest = md_dir / f"{meta['doc_id']}_p{meta['page']}.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(markdown)
    pd.DataFrame(rows).to_parquet(out / "pages.parquet", index=False)

    stats = {"pdfs": len(pdfs), "pages": len(pages), "render_seconds": render_s,
             "inference_seconds": infer_s, "pages_per_second": len(pages) / infer_s,
             "pages_finished_cleanly": finished / len(pages),
             "output_tokens_per_second": sum(r["output_tokens"] for r in rows) / infer_s}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"--- sample page {rows[0]['doc_id']} ---\n{rows[0]['markdown'][:1200]}")
    print(f"Wrote {len(rows)} Markdown pages and pages.parquet to {out}")
    with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
        mlflow.log_metrics(stats)


if __name__ == "__main__":
    main()
