# PDF to Markdown at scale with olmOCR 2 and vLLM

This recipe converts PDFs to Markdown with `allenai/olmOCR-2-7B-1025-FP8`, a document-OCR fine-tune of Qwen2.5-VL, using vLLM offline batch inference on one H100. It renders each page with `pypdfium2` at olmOCR's training resolution (1,288 px on the longest side) and sends it with the olmOCR v4 prompt. The model returns Markdown, with tables as HTML and equations as LaTeX, plus page metadata.

The input is 300 one-page PDFs (tables, multi-column papers and old scans) from the public `allenai/olmOCR-bench` dataset. The run takes 5 minutes 44 seconds in the container, plus provisioning and queueing.

## Before you run

The recipe writes its outputs to `/Volumes/main/default/air_examples/document-ocr-olmocr2-vllm`. The volume must exist and be writable. To write to another volume, override `parameters.output_root`:

```bash
databricks air run -f workload.yaml --override parameters.output_root=/Volumes/<catalog>/<schema>/<volume>/<dir> --watch
```

## Run

```bash
databricks air run -f workload.yaml --watch
```

To process your own PDFs, replace the `snapshot_download` call in `ocr.py` with a list of files, for example from a Unity Catalog volume.

## Results

The run writes these files to `<output_root>/<MLflow run ID>/`:

- `markdown/<category>/<doc>_p<page>.md`: one Markdown file per page.
- `pages.parquet`: one row per page, with the Markdown, detected language, `is_table`, `is_diagram`, rotation correction and output token count. To save it as a Delta table, run `spark.read.parquet(path).write.saveAsTable("<catalog>.<schema>.<table>")`.
- `stats.json`.

A validation run on an H100 produced these results:

| Metric | Value |
|---|---|
| Pages rendered | 300 in 41 s (CPU) |
| OCR throughput | 4.1 pages/s, 4,495 output tokens/s |
| Pages that finished cleanly (not cut off at `max_tokens`) | 98.7% |
