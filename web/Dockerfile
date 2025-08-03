FROM python:3.12-slim

WORKDIR /app

# Instala setuptools y actualiza pip ANTES de instalar otros paquetes
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "app.py"]
