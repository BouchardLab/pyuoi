#!/bin/bash

set -euo pipefail

usage() {
    echo "Usage: $0 LOGFILE"
    echo "Example: $0 Lr5a20"
}

if [[ "$#" -ne 1 ]]; then
    usage
    exit 2
fi

log="$1"
script_dir="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -f "$log" && -f "$script_dir/$log" ]]; then
    log="$script_dir/$log"
fi

if [[ ! -f "$log" ]]; then
    echo "ERROR: logfile not found: $1" >&2
    exit 1
fi

awk '
function die(msg) {
    print "ERROR: " msg > "/dev/stderr"
    exit 1
}

function wall_minutes(txt,    h, m, s, rest, a) {
    rest = txt
    sub(/^real[ \t]+/, "", rest)
    h = 0
    m = 0
    s = 0

    if (index(rest, "h") > 0) {
        split(rest, a, "h")
        h = a[1] + 0
        rest = a[2]
    }
    if (index(rest, "m") > 0 && index(rest, "s") > 0) {
        split(rest, a, "m")
        m = a[1] + 0
        s = a[2]
        sub(/s.*$/, "", s)
        return h * 60 + m + s / 60.0
    }
    if (rest ~ /^[0-9.]+s/) {
        s = rest
        sub(/s.*$/, "", s)
        return h * 60 + s / 60.0
    }
    die("cannot parse wall time line: " txt)
}

BEGIN {
    in_bags = 0
}

/^=== FDR bags:/ {
    in_bags = 1
}

/^numBags=/ && num_bags == "" {
    num_bags = $0
    sub(/^numBags=/, "", num_bags)
    sub(/[^0-9].*$/, "", num_bags)
}

/Stage \(b\): bags=[0-9]+/ && num_bags == "" {
    num_bags = $0
    sub(/^.*bags=/, "", num_bags)
    sub(/[^0-9].*$/, "", num_bags)
}

/^--- bag[ \t]+[0-9][0-9][0-9]\/[0-9][0-9][0-9]/ && num_bags == "" {
    num_bags = $0
    sub(/^.*\//, "", num_bags)
    sub(/[^0-9].*$/, "", num_bags)
    num_bags += 1
}

/^real[ \t]+[0-9]/ {
    if (!in_bags && em_min == "") {
        em_min = wall_minutes($0)
    } else if (in_bags && bag_min == "") {
        bag_min = wall_minutes($0)
    }
}

END {
    if (em_min == "") die("missing Reference EM real time")
    if (bag_min == "") die("missing first FDR bag real time")
    if (num_bags == "") die("missing requested bag count")

    total_min = em_min + num_bags * bag_min
    printf "expected_runtime_min=%.1f\n", total_min
    printf "em_min=%.1f first_bag_min=%.1f num_bags=%d\n", em_min, bag_min, num_bags
}
' "$log"
