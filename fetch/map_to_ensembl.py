#!/usr/bin/env python3
import argparse, csv, sys
from collections import defaultdict
from typing import List, Dict, Set

try:
    import mygene
except ImportError:
    print("Please install mygene first: pip install mygene", file=sys.stderr)
    sys.exit(1)

def read_gene_symbols_from_csv(path: str) -> List[str]:
    """Read the second column from a CSV file (skips empty values)."""
    syms = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        for row in r:
            if row[0].strip():
                syms.append(row[0].strip())
    return syms

def parse_list_field(s: str) -> List[str]:
    """Split 'A, B, C' → ['A','B','C']; tolerate empty/None."""
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def load_hgnc_reverse_index(path: str):
    """
    Build lookups from a tab-separated HGNC export with columns:
    'HGNC ID', 'Approved symbol', 'Status', 'Previous symbols', 'Alias symbols'
    Returns:
      - where_in_previous: symbol(lower) -> set of approved symbols that list it in Previous symbols
      - where_in_alias:    symbol(lower) -> set of approved symbols that list it in Alias symbols
      - approved_to_alias: approved symbol -> set of its alias symbols
    """
    where_in_previous: Dict[str, Set[str]] = defaultdict(set)
    where_in_alias: Dict[str, Set[str]] = defaultdict(set)
    approved_to_alias: Dict[str, Set[str]] = defaultdict(set)

    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        # tolerate alternate headers by normalizing keys
        def get(row, *names):
            for n in names:
                if n in row:
                    return row[n]
            return ""

        for row in r:
            approved = get(row, "Approved symbol", "symbol", "Approved Symbol").strip()
            prevs = parse_list_field(get(row, "Previous symbols", "prev_symbol", "Previous Symbols"))
            alias = parse_list_field(get(row, "Alias symbols", "alias_symbol", "Alias Symbols"))

            for p in prevs:
                where_in_previous[p.lower()].add(approved)
            for a in alias:
                where_in_alias[a.lower()].add(approved)
            if approved:
                approved_to_alias[approved].update(alias)

    return where_in_previous, where_in_alias, approved_to_alias

def extract_ensembl_ids(hit: dict) -> List[str]:
    """
    mygene returns 'ensembl' as dict or list of dicts; each has key 'gene'.
    Return unique list of gene IDs as strings.
    """
    ens = hit.get("ensembl")
    ids = []
    if isinstance(ens, dict):
        g = ens.get("gene")
        if g:
            ids.append(str(g))
    elif isinstance(ens, list):
        for item in ens:
            g = item.get("gene")
            if g:
                ids.append(str(g))
    return sorted(set(ids))

def main():
    ap = argparse.ArgumentParser(description="Map gene symbols to Ensembl Gene IDs using mygene and HGNC fallbacks.")
    ap.add_argument("--genes", required=True, help="Path to genes.csv (gene symbol in 2nd column).")
    ap.add_argument("--hgnc", required=True, help="Path to HGNC approved_symbols.txt (TSV).")
    ap.add_argument("--output", required=True, help="Path to output CSV.")
    args = ap.parse_args()

    # 1) Read input symbols
    gene_symbols = read_gene_symbols_from_csv(args.genes)
    if not gene_symbols:
        print("No gene symbols read from the CSV (second column).", file=sys.stderr)
        sys.exit(1)

    mg = mygene.MyGeneInfo()

    # 2) First pass: batch querymany
    batch_hits = mg.querymany(
        gene_symbols,
        scopes="symbol",
        fields="ensembl.gene",
        species="human",
        verbose=False
    )

    # Collect first-pass results keyed by original query
    first_pass: Dict[str, dict] = {}
    for h in batch_hits:
        q = h.get("query")
        if q is None:
            continue
        first_pass[q] = h

    # Prepare output
    out_fields = ["input_symbol", "resolved_symbol", "ensembl_gene_id", "source", "note"]
    out = open(args.output, "w", newline="")
    w = csv.DictWriter(out, fieldnames=out_fields)
    w.writeheader()

    total = len(gene_symbols)
    found_direct = 0
    found_via_hgnc = 0
    still_missing = 0

    # Write all first-pass results
    missing_after_first = []
    for sym in gene_symbols:
        hit = first_pass.get(sym, {})
        ids = extract_ensembl_ids(hit) if hit else []
        if ids:
            w.writerow({
                "input_symbol": sym,
                "resolved_symbol": sym,
                "ensembl_gene_id": ";".join(ids),
                "source": "direct",
                "note": ""
            })
            found_direct += 1
        else:
            # queue for HGNC fallback
            w.writerow({
                "input_symbol": sym,
                "resolved_symbol": "",
                "ensembl_gene_id": "",
                "source": "direct",
                "note": "NOT_FOUND"
            })
            missing_after_first.append(sym)

    # 3) Build HGNC helper indices
    where_prev, where_alias, approved_to_alias = load_hgnc_reverse_index(args.hgnc)

    # 4) For each missing symbol, try HGNC mapping(s) then query again (one by one)
    for sym in missing_after_first:
        candidates: Set[str] = set()

        # If the symbol appears as a previous or alias in HGNC, collect its current approved symbols
        candidates.update(where_prev.get(sym.lower(), set()))
        candidates.update(where_alias.get(sym.lower(), set()))

        # Additionally, if any approved symbols were found, also try their alias symbols (broaden search)
        broader = set()
        for ap_sym in list(candidates):
            broader.update(approved_to_alias.get(ap_sym, set()))
        candidates.update(broader)

        resolved = False
        tried = []

        for cand in sorted(candidates):
            tried.append(cand)
            h = mg.query(cand, scopes="symbol", fields="ensembl.gene", species="human")
            ids = extract_ensembl_ids(h) if isinstance(h, dict) else []
            if ids:
                w.writerow({
                    "input_symbol": sym,
                    "resolved_symbol": cand,
                    "ensembl_gene_id": ";".join(ids),
                    "source": "hgnc",
                    "note": "via previous/alias"
                })
                found_via_hgnc += 1
                resolved = True
                break

        if not resolved:
            w.writerow({
                "input_symbol": sym,
                "resolved_symbol": "",
                "ensembl_gene_id": "",
                "source": "hgnc",
                "note": f"STILL_NOT_FOUND; tried={','.join(tried) if tried else 'no_hgnc_match'}"
            })
            still_missing += 1

    out.close()

    # 5) Stats
    print("=== Summary ===")
    print(f"Total input symbols         : {total}")
    print(f"Found in first (direct) pass: {found_direct}")
    print(f"Resolved via HGNC fallback  : {found_via_hgnc}")
    print(f"Still missing               : {still_missing}")
    print(f"Output written to           : {args.output}")

if __name__ == "__main__":
    main()
