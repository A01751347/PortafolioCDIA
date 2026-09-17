#!/bin/bash
# Convierte un reporte en Markdown a PDF.
#
#   uso:  herramientas/md_a_pdf.sh M2_Analisis/reporte.md M2_Analisis/Reporte_Analisis.pdf
#
# Requiere pandoc y Google Chrome (que se usa en modo headless para imprimir).
set -euo pipefail

ENTRADA="$1"
SALIDA="$2"
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pandoc "$ENTRADA" \
  --standalone \
  --mathml \
  --embed-resources \
  --css "$AQUI/estilo.css" \
  --metadata title="" \
  --resource-path="$(cd "$(dirname "$ENTRADA")" && pwd)" \
  -o "$TMP/reporte.html"

"$CHROME" --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
  --print-to-pdf="$(cd "$(dirname "$SALIDA")" && pwd)/$(basename "$SALIDA")" \
  "file://$TMP/reporte.html" 2>/dev/null

echo "PDF generado: $SALIDA"
