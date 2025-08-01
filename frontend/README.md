# Flipkart Integrated Frontend

A single Flask server that integrates both autosuggest and search functionality with a modern, responsive design.

## Features

- **Smart Autosuggest**: Real-time suggestions as you type
- **Advanced Search**: Full product search with semantic matching
- **Dual View Modes**: Grid and list views for search results
- **Mobile Responsive**: Optimized for all screen sizes
- **Contextual**: Persona, location, and event-based suggestions
- **Modern UI**: Clean, professional design with smooth animations
- **Single Server**: Everything runs on one port - no microservices needed!

## Architecture

The integrated server:
1. Loads autosuggest system directly into memory
2. Loads search system (NER + FAISS + reranking) directly
3. Serves the web application
4. Provides all APIs in a single Flask server
5. **No external services required!**

## Prerequisites

**Nothing!** Just install dependencies and run the single server.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the integrated server (that's it!):
   ```bash
   python app.py
   ```

3. Access the application at: http://localhost:3000

## API Endpoints

### Integrated Endpoints
- `GET /` - Main application
- `GET /api/health` - Health check with component status
- `GET /api/config` - Configuration data
- `POST /api/suggest` - Integrated autosuggest (no proxy)
- `POST /api/search` - Integrated search (no proxy)
- `GET /api/analytics` - System analytics

### Usage Examples

**Autosuggest Request:**
```json
POST /api/suggest
{
    "query": "laptop",
    "persona": "tech_enthusiast",
    "location": "Mumbai",
    "event": "none",
    "max_suggestions": 5
}
```

**Search Request:**
```json
POST /api/search
{
    "query": "laptop samsung",
    "context": {
        "location": "Mumbai",
        "persona": "tech_enthusiast"
    },
    "top_k": 20
}
```

## File Structure

```
frontend/
├── app.py              # Unified Flask server
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── templates/
│   └── index.html     # Main HTML template
└── static/
    ├── style.css      # Comprehensive CSS styles
    └── script.js      # JavaScript application logic
```

## Key Features Implementation

### Autosuggest Integration
- Real-time suggestions with 200ms debounce
- Keyboard navigation (arrow keys, enter, escape)
- Click-to-select functionality
- Context-aware suggestions based on persona, location, event

### Search Results
- Grid view: Card-based layout with product images
- List view: Detailed horizontal layout
- Responsive design for mobile devices
- Loading states and error handling
- No results messaging

### Mobile Responsiveness
- Optimized layouts for tablets and phones
- Touch-friendly interface elements
- Responsive grid that adapts to screen size
- Mobile-first CSS approach

### Error Handling
- Graceful degradation when services are unavailable
- User-friendly error messages
- Fallback configurations
- Service health monitoring

## Development

### Running in Development Mode
```bash
python app.py --host 0.0.0.0 --port 3000
```

### Production Mode
```bash
python app.py --no-debug --host 0.0.0.0 --port 3000
```

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Performance

- All systems loaded into memory for fast response times
- Optimized CSS with CSS Grid and Flexbox
- Debounced API calls to reduce server load
- Efficient DOM manipulation
- Responsive images and layouts
- Single server architecture reduces network overhead

## Troubleshooting

### System Components Not Loading
If autosuggest or search systems fail to load:
1. Check that all required files exist in `autosuggest/` and `searchresultpage/` directories
2. Install missing dependencies: `pip install -r requirements.txt`
3. Check console output for specific error messages
4. Ensure datasets are in the `dataset/` directory

### Port Conflicts
If port 3000 is in use:
```bash
python app.py --port 3001
```

### Import Errors
If you see import errors:
1. Make sure you're running from the `frontend/` directory
2. Install all dependencies: `pip install -r requirements.txt`
3. Check that `autosuggest/` and `searchresultpage/` directories exist

### Memory Issues
The integrated server loads large models. If you encounter memory issues:
1. Ensure you have at least 4GB available RAM
2. Close other applications
3. Consider running with `--no-debug` to reduce memory usage

### CORS Issues
The server includes CORS headers. If you still encounter issues, check your browser's developer console.