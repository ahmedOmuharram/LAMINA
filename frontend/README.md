# Materials Science AI Frontend

A modern React + TypeScript frontend for interacting with the Materials Project AI backend, built to replace OpenWebUI with a custom, streamlined interface.

## Tech Stack

- **React 19** with **TypeScript**
- **Vite 7** for fast development
- **Tailwind CSS v4** for styling
- **ShadCN UI** for beautiful components
- **Server-Sent Events (SSE)** for real-time streaming

## Setup

```bash
npm install
npm run dev
```

The frontend will run on `http://localhost:3000` and proxy API requests to the backend at `http://localhost:8000`.

## Key Features

### Core Functionality
- ✅ Real-time chat interface with streaming support
- ✅ Multiple model selection (GPT-4.1, o1, GPT-4o-mini)
- ✅ Image upload and multimodal support
- ✅ Interactive plot display
- ✅ Conversation history management

### Analytics & Monitoring
- ✅ Tool usage tracking
- ✅ LLM usage metrics
- ✅ Response time analytics
- ✅ Token consumption dashboard

### Materials Science Integration
- ✅ CALPHAD phase diagram tools
- ✅ Electrochemistry calculations
- ✅ Materials search integration
- ✅ Interactive plot viewing

## Project Structure

```
frontend/
├── src/
│   ├── components/         # React components
│   │   ├── ui/            # ShadCN UI components
│   │   ├── chat/          # Chat interface components
│   │   ├── analytics/     # Analytics dashboard
│   │   └── layout/        # Layout components
│   ├── lib/               # Utilities and helpers
│   │   ├── api.ts         # API client
│   │   ├── streaming.ts   # SSE streaming utilities
│   │   └── utils.ts       # General utilities
│   ├── types/             # TypeScript type definitions
│   ├── hooks/             # Custom React hooks
│   └── App.tsx            # Main application component
├── public/                # Static assets
└── index.html             # Entry HTML file
```

## Development Tasks

### Phase 1: Core Setup ✅
- [x] Initialize Vite + React + TypeScript
- [x] Configure Tailwind CSS v4
- [x] Setup ShadCN UI
- [x] Configure path aliases

### Phase 2: API & Streaming 🚧
- [ ] Create API client with fetch
- [ ] Implement SSE streaming handler
- [ ] Setup message type definitions
- [ ] Add error handling

### Phase 3: Chat Interface 🚧
- [ ] Build message list component
- [ ] Create message input component
- [ ] Add streaming text rendering
- [ ] Implement tool call display
- [ ] Add image upload UI

### Phase 4: Advanced Features 📋
- [ ] Model selection dropdown
- [ ] Analytics dashboard
- [ ] Conversation history
- [ ] Plot viewer component
- [ ] Export functionality

## API Integration

The frontend communicates with the backend via:

- `POST /v1/chat/completions` - Main chat endpoint (streaming & non-streaming)
- `GET /v1/models` - List available models
- `GET /static/plots/*` - Retrieve generated plots

## Configuration

Environment variables (`.env`):
```
VITE_API_URL=http://localhost:8000
```

## Notes

- This frontend is designed to work seamlessly with the existing FastAPI backend
- SSE streaming is handled natively with `fetch` and `ReadableStream`
- All components are built with accessibility in mind
- The design follows materials science research workflows
