# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory 
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the rest of the app code to the working directory
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run the Streamlit app
CMD streamlit run app.py