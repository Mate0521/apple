FROM python:3.11-slim

WORKDIR /app

COPY . .

# Instalamos las librerías necesarias para el proyecto
# flask -> servidor web
# pandas -> manipulación de datos
# matplotlib y seaborn -> gráficas para el EDA
RUN pip install flask pandas matplotlib seaborn

CMD ["python", "src/main.py"]