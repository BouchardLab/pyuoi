#!/bin/bash

set -e

doc_name=nonStation-ver5_synDeprGen

is_ubuntu_or_macos=0
if [ -f /etc/os-release ] && grep -qi 'ubuntu' /etc/os-release; then
    is_ubuntu_or_macos=1
elif [ "$(uname -s)" = "Darwin" ]; then
    is_ubuntu_or_macos=1
fi

if ! command -v pdflatex >/dev/null 2>&1; then
    module load texlive
fi

pdflatex -interaction=nonstopmode -halt-on-error "$doc_name"
pdflatex -interaction=nonstopmode -halt-on-error "$doc_name"

pdf_file="${doc_name}.pdf"
echo "Built $pdf_file"

if [ "$is_ubuntu_or_macos" -eq 0 ]; then
    gv "$pdf_file"
fi
