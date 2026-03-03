# R.A.D.H.A - Responsive And Deeply Human Assistant

A powerful AI assistant built with FastAPI and LangChain, featuring real-time responses, semantic search, and intelligent chat memory.

## Features

- **AI-Powered Chat**: Powered by Groq's LLaMA 3.3 70B model for fast, intelligent responses
- **Real-Time Search**: Integration with Tavily for current web search capabilities
- **Vector Store**: FAISS-based semantic search with sentence transformers for smart response generation
- **Chat Memory**: Persistent chat history with contextual awareness
- **Web Interface**: Clean, responsive frontend built with HTML/CSS/JavaScript
- **REST API**: FastAPI backend for easy integration

## Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Shubham1905s/navAjna.git
cd Radha
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
ASSISTANT_NAME=Radha
```

5. **Run the project**
```bash
python run.py
```

The application will start on `http://localhost:7860`

## Access Points

- **API**: http://localhost:7860
- **Frontend**: http://localhost:7860/app/
- **Health Check**: http://localhost:7860/health

## API Endpoints

### Chat Endpoints
- `POST /chat` - Send a message and get a response
- `GET /chat/{chat_id}` - Get chat history
- `DELETE /chat/{chat_id}` - Delete a chat

### Search Endpoints
- `POST /search` - Perform web search with real-time results

### Health
- `GET /health` - Check API status

## Project Structure

```
Radha/
├── app/
│   ├── main.py              # FastAPI application
│   ├── models.py            # Data models
│   └── services/
│       ├── chat_service.py  # Chat logic
│       ├── groq_service.py  # Groq AI integration
│       ├── realtime_service.py  # Real-time search
│       └── vector_store.py  # Vector store & embeddings
├── frontend/                # Web interface
├── database/
│   ├── chats_data/         # Chat history
│   ├── learning_data/      # Custom knowledge base
│   └── vector_store/       # FAISS index
├── config.py               # Configuration
├── run.py                  # Entry point
└── requirements.txt        # Dependencies
```

## Configuration

Edit `config.py` to customize:
- Model selection
- Chunk size and overlap for vector store
- Maximum chat history length
- System prompt and assistant personality

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API key | Yes |
| `TAVILY_API_KEY` | Tavily search API key | Yes |
| `GROQ_MODEL` | Model to use (default: llama-3.3-70b-versatile) | No |
| `ASSISTANT_NAME` | Assistant name (default: Radha) | No |
| `TTS_VOICE` | Text-to-speech voice | No |

## Docker

Build and run with Docker:

```bash
docker build -t radha .
docker run -p 7860:7860 --env-file .env radha
```

## Technologies Used

- **FastAPI** - Modern Python web framework
- **LangChain** - LLM orchestration framework
- **Groq** - Fast AI inference
- **FAISS** - Vector similarity search
- **Sentence Transformers** - Semantic embeddings
- **Tavily** - Web search API
- **Uvicorn** - ASGI web server

## License

MIT License - feel free to use this project for your own purposes.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Made with ❤️ by Shubham1905s**
