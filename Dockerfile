# 1) Base image with Python 3.11.2
FROM python:3.11.2

# 2) Install OS-level dependencies
#    PyTorch CPU wheels typically work out of the box,
#    but we install a couple of common libraries just in case.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libatlas-base-dev \
      libjpeg-dev \
      cmake\
 && rm -rf /var/lib/apt/lists/*

# 3) Create and switch to a non-root user (optional, but recommended)
RUN useradd --create-home appuser
WORKDIR /home/appuser

# 4) Copy dependency files and install Python packages
COPY requirements.txt .

# Use pip to install exactly what's in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --user

# 5) Copy your application code
COPY face.py .

# 6) Expose the port your Flask app listens on
EXPOSE 5000

# 7) Run as non-root user
USER appuser

# 8) Default command to launch your app
CMD ["python", "face.py"]
