#!/usr/bin/env bash

set -euo pipefail

# INTEGER:BASENAME entries. The .tex suffix is added by the script.
document_map=(
    1:genDale_topo_states_ver3c
    2:fitFDR_states_ver3c
    3:fitEM_states_ver3c
    4:deBiasFit_states_ver3c
    5:edgeMeterStability_ver3c
    6:edgeMeterAccuracy_ver3c
    10:preproc_exper_ver3c
)

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp_dir="${script_dir}/tmp"

usage() {
    local entry

    printf 'Usage: %s NUMBER|all|--list\n\n' "${0##*/}"
    printf 'Documents:\n'
    for entry in "${document_map[@]}"; do
        printf '  %3s  %s.tex\n' "${entry%%:*}" "${entry#*:}"
    done
}

ensure_pdflatex() {
    if command -v pdflatex >/dev/null 2>&1; then
        return
    fi
    if type module >/dev/null 2>&1; then
        module load texlive
    fi
    if ! command -v pdflatex >/dev/null 2>&1; then
        echo "ERROR: pdflatex is unavailable; load a TeX Live environment." >&2
        exit 1
    fi
}

build_document() {
    local doc_name=$1
    local tex_file="${doc_name}.tex"
    local tmp_pdf="${tmp_dir}/${doc_name}.pdf"

    if [ ! -f "$tex_file" ]; then
        echo "ERROR: input file not found: ${script_dir}/${tex_file}" >&2
        return 1
    fi

    echo "Building ${tex_file}"
    pdflatex -interaction=nonstopmode -halt-on-error \
        -output-directory="$tmp_dir" "$tex_file"
    pdflatex -interaction=nonstopmode -halt-on-error \
        -output-directory="$tmp_dir" "$tex_file"
    mv -f "$tmp_pdf" "${script_dir}/${doc_name}.pdf"
    echo "Built ${doc_name}.pdf (auxiliary files: tmp/)"
}

if [ "$#" -eq 0 ] || { [ "$#" -eq 1 ] && [ "$1" = "--list" ]; }; then
    usage
    exit 0
fi
if [ "$#" -ne 1 ]; then
    echo "ERROR: expected one document number, 'all', or '--list'." >&2
    usage >&2
    exit 2
fi

cd "$script_dir"
mkdir -p "$tmp_dir"
ensure_pdflatex

if [ "$1" = "all" ]; then
    for entry in "${document_map[@]}"; do
        build_document "${entry#*:}"
    done
    exit 0
fi

doc_name=
for entry in "${document_map[@]}"; do
    if [ "$1" = "${entry%%:*}" ]; then
        doc_name=${entry#*:}
        break
    fi
done
if [ -z "$doc_name" ]; then
    echo "ERROR: unknown document number: $1" >&2
    usage >&2
    exit 2
fi

build_document "$doc_name"
