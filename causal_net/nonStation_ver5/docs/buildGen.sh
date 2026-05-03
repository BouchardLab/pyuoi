#!/bin/bash

set -e

doc_name=nonStation-ver5_synDeprGen

is_ubuntu=0
if [ -f /etc/os-release ] && grep -qi 'ubuntu' /etc/os-release; then
    is_ubuntu=1
fi
 echo is_ubuntu=$is_ubuntu

if [ "$is_ubuntu" -eq 1 ]; then
    pdflatex "$doc_name"
    pdflatex "$doc_name"
#    open "${doc_name}.pdf"
else
    if ! command -v latex >/dev/null 2>&1; then
        module load texlive
    fi
    #latex "$doc_name"
    #latex "$doc_name"
    pdflatex "$doc_name"  # to produce .pdf for github
 
#    xdvi -s 5 "${doc_name}.dvi"
fi
