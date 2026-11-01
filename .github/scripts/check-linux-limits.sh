#!/usr/bin/env bash
# Usage: check-linux-limits.sh <file> <max GLIBC_ version> <max GLIBCXX_ version>
set -euo pipefail

file="$1"
max_glibc="$2"
max_glibcxx="$3"

highest() {
    objdump -T "$file" | grep -o "$1_[0-9][0-9.]*" | sed "s/$1_//" | sort -V | tail -1
}

glibc=$(highest GLIBC)
glibcxx=$(highest GLIBCXX || true)
echo "$(basename "$file"): GLIBC_${glibc:-none} (max $max_glibc), GLIBCXX_${glibcxx:-none} (max $max_glibcxx)"

over() {
    [ -n "$1" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" != "$2" ]
}

if over "$glibc" "$max_glibc" || over "$glibcxx" "$max_glibcxx"; then
    echo "::error file=$file::symbol versions exceed the shipped binary's"
    exit 1
fi
