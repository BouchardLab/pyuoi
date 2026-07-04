#!/bin/bash

set -e

doc_name=deBiasFit_states_ver3c

if ! command -v pdflatex >/dev/null 2>&1; then
    module load texlive
fi

pdflatex -interaction=nonstopmode -halt-on-error "$doc_name"
pdflatex -interaction=nonstopmode -halt-on-error "$doc_name"

pdf_file="${doc_name}.pdf"
echo "Built $pdf_file"

# Open the PDF based on the specific environment
if [ "$(uname -s)" = "Darwin" ]; then
    open "$pdf_file"
elif [ -f /etc/os-release ] && grep -qi 'ubuntu' /etc/os-release; then
    xdg-open "$pdf_file"
elif hostname | grep -qiE 'login|nid0'; then
    gv "$pdf_file"
else
    echo "Environment not matched. PDF is ready at $pdf_file"
fi
