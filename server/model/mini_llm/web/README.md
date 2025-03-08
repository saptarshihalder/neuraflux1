# MiniLLM Math Chat Interface

A web-based chat interface for interacting with the MiniLLM mathematical problem-solving model.

## Features

- **ChatGPT-style Interface**: Familiar and intuitive chat experience
- **Real-time Responses**: Quick feedback from the model
- **Step-by-step Solutions**: Option to view detailed solution steps
- **Mathematical Rendering**: Proper rendering of mathematical expressions using KaTeX
- **Example Problems**: Pre-populated examples for easy testing

## Setup

### Prerequisites

- Python 3.8+
- Flask
- The MiniLLM model and dependencies

### Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Web Interface

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Configuration

You can configure the following environment variables to customize the behavior:

- `MODEL_PATH`: Path to the fine-tuned model (default: `output/math_finetuned/final_model`)
- `TOKENIZER_PATH`: Path to the tokenizer (default: `output/math_finetuned/final_model/tokenizer`)
- `PORT`: Port to run the server on (default: 5000)

## Deployment

For production deployment, consider:

1. Using a production WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 app:app
   ```

2. Adding HTTPS support with a reverse proxy like Nginx

## Project Structure

```
web/
├── app.py              # Flask application
├── requirements.txt    # Package dependencies
├── static/             # Static assets
│   ├── css/            # Stylesheets
│   │   └── styles.css  # Main stylesheet
│   └── js/             # JavaScript files
│       └── script.js   # Client-side logic
└── templates/          # HTML templates
    └── index.html      # Main chat interface
``` 