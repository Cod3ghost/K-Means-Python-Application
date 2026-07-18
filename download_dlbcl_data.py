"""
download_dlbcl_data.py

Downloads open-access RNA-Seq gene expression quantification files for the
"National Cancer Institute Center for Cancer Research - Diffuse Large B
Cell Lymphoma (DLBCL) Genomics and Expression" project (NCICCR-DLBCL) from
the Genomic Data Commons (GDC) API. This project is registered on the AWS
Registry of Open Data (arn:aws:s3:::gdc-nciccr-phs001444-2-open), but GDC's
own HTTPS API serves the same open-tier files without needing AWS
credentials, the AWS CLI, or an AWS account, which makes it a much simpler
download path.

This script uses ONLY the Python standard library (urllib, json, csv) so it
runs with no pip installs, straight out of the box in PyCharm.

What it does
------------
1. Queries the GDC "files" endpoint for every file in project NCICCR-DLBCL
   that is: data_type = "Gene Expression Quantification" AND access = "open"
   (these are per-sample "*.rna_seq.augmented_star_gene_counts.tsv" files,
   the STAR-based gene count/TPM/FPKM output GDC produces for each sample;
   everything else in this project, aligned BAM files, fusion calls, etc.,
   is controlled-access and is deliberately skipped).
2. Writes a manifest (data/manifest.csv) mapping each file's GDC UUID to
   its originating case (patient) and sample submitter ID.
3. Downloads each file to data/raw/<file_id>.tsv via GDC's public
   /data/<file_id> endpoint, skipping files already downloaded so the
   script is safe to re-run if interrupted.

Run this from a machine with normal internet access (this script cannot
run inside the sandboxed environment used to build this application,
which has no outbound access to GDC or AWS).

Usage:
    python download_dlbcl_data.py [--limit N] [--data-dir ../data]

    --limit N     Only download the first N files (useful for a quick
                  test run before committing to the full ~500-file,
                  ~2 GB download). Omit to download everything available.

Author: Desmond Amos Bature
Course: BAN6440 - Applied Machine Learning for Business Analytics
Module 4 Assignment - K-Means Python Application
"""

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
PROJECT_ID = "NCICCR-DLBCL"
PAGE_SIZE = 100


def fetch_manifest():
    """Page through the GDC files endpoint and return the list of open
    Gene Expression Quantification files for the DLBCL project."""
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [PROJECT_ID]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ],
    }
    fields = "file_id,file_name,file_size,cases.case_id,cases.submitter_id,cases.samples.sample_type"

    manifest = []
    from_offset = 0
    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": fields,
            "size": str(PAGE_SIZE),
            "from": str(from_offset),
            "format": "JSON",
        }
        url = f"{GDC_FILES_ENDPOINT}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        hits = payload["data"]["hits"]
        total = payload["data"]["pagination"]["total"]

        for hit in hits:
            cases = hit.get("cases", [{}])
            case = cases[0] if cases else {}
            samples = case.get("samples", [{}])
            sample_type = samples[0].get("sample_type") if samples else None
            manifest.append({
                "file_id": hit["file_id"],
                "file_name": hit["file_name"],
                "file_size": hit.get("file_size"),
                "case_id": case.get("case_id"),
                "case_submitter_id": case.get("submitter_id"),
                "sample_type": sample_type,
            })

        from_offset += PAGE_SIZE
        print(f"  ... fetched {min(from_offset, total)} / {total} file records")
        if from_offset >= total:
            break

    return manifest


def download_file(file_id, dest_path, retries=3):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return "skipped"

    url = f"{GDC_DATA_ENDPOINT}/{file_id}"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(dest_path, "wb") as out:
                out.write(resp.read())
            return "downloaded"
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries:
                return f"failed: {e}"
            time.sleep(2 * attempt)


def main():
    parser = argparse.ArgumentParser(description="Download open DLBCL RNA-Seq gene count files from GDC")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only download the first N files (for a quick test run)")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    args = parser.parse_args()

    raw_dir = os.path.join(args.data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print(f"Querying GDC API for open Gene Expression Quantification files in project {PROJECT_ID} ...")
    manifest = fetch_manifest()
    print(f"Found {len(manifest)} open, per-sample gene expression files.")

    if args.limit is not None:
        manifest = manifest[:args.limit]
        print(f"Limiting to the first {len(manifest)} files (--limit {args.limit}).")

    manifest_path = os.path.join(args.data_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_id", "file_name", "file_size", "case_id",
                                                "case_submitter_id", "sample_type"])
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Manifest written to {manifest_path}")

    total_bytes = sum(m["file_size"] or 0 for m in manifest)
    print(f"Downloading {len(manifest)} files (~{total_bytes / 1e9:.2f} GB total) to {raw_dir} ...")

    downloaded = skipped = failed = 0
    for i, m in enumerate(manifest, start=1):
        dest = os.path.join(raw_dir, f"{m['file_id']}.tsv")
        result = download_file(m["file_id"], dest)
        if result == "downloaded":
            downloaded += 1
        elif result == "skipped":
            skipped += 1
        else:
            failed += 1
            print(f"  [{i}/{len(manifest)}] FAILED {m['file_id']}: {result}")

        if i % 25 == 0 or i == len(manifest):
            print(f"  ... {i}/{len(manifest)} processed (downloaded={downloaded}, skipped={skipped}, failed={failed})")

    print("Done.")
    print(f"Downloaded: {downloaded}, already present: {skipped}, failed: {failed}")
    print(f"Raw files:  {raw_dir}")
    print(f"Manifest:   {manifest_path}")


if __name__ == "__main__":
    main()
