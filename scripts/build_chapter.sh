#!/usr/bin/env bash
# Build a single chapter in isolation, e.g.
#   bash scripts/build_chapter.sh 03-diffusion
# Output: build/<chapter>/test.pdf  (never touches main.aux / main.pdf)
set -euo pipefail
cd "$(dirname "$0")/.."
ch="$1"
prefix="${ch%%-*}"
dir="build/$ch"
mkdir -p "$dir"
case "$prefix" in
  A) setup='\appendix\setcounter{chapter}{0}' ;;
  B) setup='\appendix\setcounter{chapter}{1}' ;;
  *) setup="\\setcounter{chapter}{$((10#$prefix - 1))}" ;;
esac
cat > "$dir/test.tex" <<TEX
\\documentclass[11pt,a4paper,oneside]{report}
\\input{../../preamble}
\\graphicspath{{../../figures/}}
\\begin{document}
$setup
\\input{../../chapters/$ch}
\\bibliographystyle{plainnat}
\\bibliography{../../references}
\\end{document}
TEX
cd "$dir"
latexmk -pdf -interaction=nonstopmode -halt-on-error test.tex > build.log 2>&1 || {
  echo "BUILD FAILED -- last lines of test.log:"; tail -40 test.log; exit 1; }
echo "OK: $dir/test.pdf ($(pdfinfo test.pdf | awk '/Pages/{print $2}') pages)"
grep -E "LaTeX Warning: (Reference|Citation)" test.log | sed 's/^/  /' | sort -u | head -40 || true
