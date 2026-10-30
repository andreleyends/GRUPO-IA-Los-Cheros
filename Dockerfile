# StyleMate · Sprint 3 · Imagen de la app web (Diego · MLOps)
# Contexto de build: RAIZ del repo.   docker build -t stylemate:local .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HOME=/home/app \
    STYLEMATE_BACKEND=real \
    STYLEMATE_DATA=/app/sprint2/02_DATASET \
    STYLEMATE_MODELS=/app/sprint2/05_CODIGO/models \
    STYLEMATE_RULES=/app/sprint1/E4_reglas_gusto/rules.py

RUN useradd --create-home --home-dir /home/app app
WORKDIR /app

# 1) torch en version CPU (la imagen pesa ~1 GB menos que con CUDA)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) resto de dependencias (esta capa se reutiliza mientras requirements.txt no cambie)
COPY sprint3/requirements.txt sprint3/requirements.txt
RUN pip install -r sprint3/requirements.txt

# 3) codigo. Datos, modelos y rules.py NO van en la imagen: se montan (ver docker-compose.yml)
COPY sprint3/ sprint3/

# --- ESTAS SON LAS LÍNEAS NUEVAS QUE DEBES AGREGAR ---
COPY sprint1/ sprint1/
COPY sprint2/02_DATASET/ sprint2/02_DATASET/
COPY sprint2/05_CODIGO/models/ sprint2/05_CODIGO/models/
# -----------------------------------------------------

RUN chown -R app:app /app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status==200 else 1)"

CMD ["streamlit", "run", "sprint3/app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]