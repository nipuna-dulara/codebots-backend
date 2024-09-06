# codebots-backend

## Overview

This FastAPI application provides a chatbot service to assist tourists visiting Sri Lanka. The chatbot is designed to answer questions about local attractions, rules, regulations, and other travel-related queries. It uses a generative AI model to interact with users and provide information in a friendly and formal manner.

## Features

- **Chat with the Bot**: Users can interact with the chatbot to get information about Sri Lanka.
- **Generative AI Integration**: Utilizes the Gemini-1.5-Flash model to generate responses based on user input.
- **Static Files**: Serve static files (e.g., images) for enhanced user experience.
- **CORS Support**: Configured to allow cross-origin requests for development purposes.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nipuna-dulara/codebots-backend.git
   cd codebots-backend
2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
    source venv/bin/activate 
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
3. **Start the server**:
   ```bash
   uvicorn main:app  --port 8000 --reload


