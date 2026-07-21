📂 Project Structure

Code
medical_AI/
│
├── Ai_ui.py                   # Graphical user interface
├── main.py                    # Terminal-based interaction
├── datascraper.py             # Medical data scraper
├── trainningforgithub.py      # Model training pipeline
│
├── mts_train_mistral_rtx4070.jsonl   # Training dataset
├── mts_valid_mistral_rtx4070.jsonl   # Validation dataset
├── valid_fixed.jsonl                 # Cleaned validation set
│
└── README.md                  # Project documentation
🛠️ Tech Stack
Python 3.10+

Ollama (model training & inference)

JSONL datasets

Custom Python training scripts

PyQt5 (UI)

GPU acceleration (RTX 4070)

Git/GitHub

🚀 Installation
1. Clone the repository
Code
git clone https://github.com/Opm-dlta/medical_AI
cd medical_AI
2. Install dependencies
Code
pip install -r requirements.txt
3. Install Ollama
Download from:
https://ollama.com

4. Run the graphical interface
Code
python Ai_ui.py
📘 Training the Model
To retrain or fine‑tune the model:

Code
python trainningforgithub.py
This script:

Loads JSONL datasets

Prepares training batches

Fine‑tunes the model using Ollama

Saves checkpoints locally

💬 Example Interaction
User:  
I have a sore throat and mild fever. What should I do?

AI:  
Based on your symptoms, this may be a viral infection. You can manage it at home with hydration, rest, and warm saltwater gargles. Seek medical attention if symptoms worsen or persist beyond 48–72 hours.

🔒 Safety Disclaimer
This AI is not a replacement for professional medical advice.
It is intended only for simple, non‑emergency guidance.

📈 Future Improvements
Upgrade to larger models (Llama 3, Mistral 8×7B)

Expand dataset with more medical dialogues

Improve UI design and add dark mode

Add more advanced safety filters

Deploy as a web or mobile app

👤 Author
Luong Anh Duc
