import os
import subprocess
from pyngrok import ngrok, conf

# --- Configure ngrok ---
conf.get_default().ngrok_path = r"C:\Users\hp\Documents\visionx dataset\titanic_project\titanic game\ngrok.exe"
conf.get_default().auth_token = "31S0uhBHod8dWKCL7xaSQiUwNPQ_2Dy14NiE4tmWnXGn5qV8R"

# Streamlit runs on port 8501 by default
public_url = ngrok.connect(8501)
print("🌍 Public URL:", public_url)

# --- Run Streamlit app ---
# Replace 'app.py' with the filename of your Streamlit app
subprocess.run(["streamlit", "run", "titanic_game_py.py"])
