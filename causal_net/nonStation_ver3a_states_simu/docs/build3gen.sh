#!/bin/bash

set -e

doc_name=genDale_topo_states_ver3
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

if ! command -v pdflatex >/dev/null 2>&1; then
    module load texlive
fi

pdflatex -interaction=nonstopmode -halt-on-error "$doc_name"
pdflatex -interaction=nonstopmode -halt-on-error "$doc_name"

pdf_file="${doc_name}.pdf"
echo "Built $script_dir/$pdf_file"

# Open the PDF based on the specific environment.
if [ "$(uname -s)" = "Darwin" ]; then
    open "$pdf_file" || echo "Could not open PDF automatically. PDF is ready at $script_dir/$pdf_file"
elif [ -f /etc/os-release ] && grep -qi 'ubuntu' /etc/os-release; then
    xdg-open "$pdf_file" || echo "Could not open PDF automatically. PDF is ready at $script_dir/$pdf_file"
elif hostname | grep -qiE 'login|nid0'; then
    if [ -n "${DISPLAY:-}" ] && command -v gv >/dev/null 2>&1; then
        gv "$pdf_file" || echo "Could not open PDF automatically. PDF is ready at $script_dir/$pdf_file"
    else
        echo "No graphical display found. PDF is ready at $script_dir/$pdf_file"
    fi
else
    echo "Environment not matched. PDF is ready at $script_dir/$pdf_file"
fi
