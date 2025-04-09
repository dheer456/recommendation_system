FROM python:3.9-slim
 
 WORKDIR /app
 
 COPY requirements.txt .
 RUN pip install --no-cache-dir -r requirements.txt
 
 COPY app.py .
 
 # Set environment variables
 ENV PYTHONUNBUFFERED=1
 ENV GRADIO_SERVER_NAME=0.0.0.0
 ENV GRADIO_SERVER_PORT=5000
 
 # Expose the port
 EXPOSE 5000
 
 # Run the application
 CMD ["python", "app.py"]
