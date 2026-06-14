# Base image — Python 3.9 slim
FROM python:3.9

# Set container working directory
WORKDIR /app

# Copy project files into the container
COPY . .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Create a non-root user for security (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Hugging Face Docker Spaces expose port 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
