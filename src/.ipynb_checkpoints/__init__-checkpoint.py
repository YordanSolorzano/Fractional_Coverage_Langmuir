# Import Library
import os

# Create a folder "results" if there was created 
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))

# Create try-except strcuture to avoid errors
try:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Directorio '{RESULTS_DIR}' creado o ya existente.")
except Exception as e:
    print(f"Error al crear el directorio '{RESULTS_DIR}': {e}")
