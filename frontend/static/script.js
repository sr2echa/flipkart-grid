let config = {};
let currentQuery = '';
let debounceTimer = null;
let currentView = 'grid';

// Initialize app
async function init() {
    try {
        // Load configuration
        const response = await fetch('/api/config');
        config = await response.json();
        
        // Populate dropdowns
        populatePersonas();
        populateLocations();
        populateEvents();
        
        // Setup event listeners
        setupEventListeners();
        
        updateStatus('Ready');
    } catch (error) {
        console.error('Initialization error:', error);
        updateStatus('Error loading configuration');
    }
}

function populatePersonas() {
    const select = document.getElementById('persona-select');
    Object.entries(config.personas).forEach(([id, persona]) => {
        const option = document.createElement('option');
        option.value = id;
        option.textContent = persona.name;
        if (id === 'tech_enthusiast') option.selected = true;
        select.appendChild(option);
    });
    updatePersonaInfo();
}

function populateLocations() {
    const select = document.getElementById('location-select');
    config.locations.forEach(location => {
        const option = document.createElement('option');
        option.value = location;
        option.textContent = location;
        if (location === 'Mumbai') option.selected = true;
        select.appendChild(option);
    });
}

function populateEvents() {
    const select = document.getElementById('event-select');
    config.events.forEach(event => {
        const option = document.createElement('option');
        option.value = event.id;
        option.textContent = event.name;
        if (event.id === 'none') option.selected = true;
        select.appendChild(option);
    });
}

function updatePersonaInfo() {
    const personaId = document.getElementById('persona-select').value;
    const persona = config.personas[personaId];
    const infoDiv = document.getElementById('persona-info');
    infoDiv.textContent = persona ? persona.description : '';
}

function setupEventListeners() {
    const searchInput = document.getElementById('search-input');
    const personaSelect = document.getElementById('persona-select');
    const locationSelect = document.getElementById('location-select');
    const eventSelect = document.getElementById('event-select');

    // Search input with debouncing
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim();
        currentQuery = query;
        
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            if (currentQuery === query) {
                performAutosuggest(query);
            }
        }, 200);
    });

    // Enter key for search
    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && currentQuery.trim()) {
            performSearch(currentQuery.trim());
        }
    });

    // Settings changes
    [personaSelect, locationSelect, eventSelect].forEach(select => {
        select.addEventListener('change', () => {
            if (select === personaSelect) {
                updatePersonaInfo();
            }
            if (currentQuery) {
                performAutosuggest(currentQuery);
            }
        });
    });

    // Suggestion clicks
    document.getElementById('suggestions-list').addEventListener('click', (e) => {
        const suggestionItem = e.target.closest('.suggestion-item');
        if (suggestionItem) {
            const suggestionText = suggestionItem.querySelector('.suggestion-text').textContent;
            searchInput.value = suggestionText;
            currentQuery = suggestionText;
            performSearch(suggestionText);
        }
    });

    // View toggle
    document.getElementById('grid-view').addEventListener('click', () => switchView('grid'));
    document.getElementById('list-view').addEventListener('click', () => switchView('list'));
}

async function performAutosuggest(query) {
    if (!query) {
        showEmptyState();
        hideSearchResults();
        return;
    }

    showLoading();

    try {
        const requestData = {
            query: query,
            persona: document.getElementById('persona-select').value,
            location: document.getElementById('location-select').value,
            event: document.getElementById('event-select').value
        };

        const response = await fetch('/api/suggest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        const data = await response.json();
        
        if (response.ok) {
            displaySuggestions(data.suggestions, data.metadata);
            updateStatus(`${data.suggestions.length} suggestions in ${data.response_time_ms}ms`);
        } else {
            showError(data.error || 'Search failed');
            updateStatus('Search failed');
        }
    } catch (error) {
        console.error('Search error:', error);
        showError('Connection error');
        updateStatus('Connection error');
    }
}

async function performSearch(query) {
    if (!query) return;

    hideSuggestions();
    showSearchResults();
    showSearchLoading();

    try {
        const requestData = {
            query: query,
            context: {
                location: document.getElementById('location-select').value,
                persona: document.getElementById('persona-select').value,
                event: document.getElementById('event-select').value
            },
            top_k: 20
        };

        const response = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        const results = await response.json();
        
        if (response.ok && Array.isArray(results)) {
            displaySearchResults(results, query);
            updateStatus(`Found ${results.length} products`);
        } else {
            showNoResults();
            updateStatus('No products found');
        }
    } catch (error) {
        console.error('Search error:', error);
        showNoResults();
        updateStatus('Search failed');
    }
}

function showLoading() {
    document.getElementById('loading').classList.add('show');
    document.getElementById('suggestions-list').innerHTML = '<div class="loading show">Searching...</div>';
}

function showSearchLoading() {
    document.getElementById('products-container').innerHTML = '<div class="loading show">Searching for products...</div>';
    document.getElementById('no-results').style.display = 'none';
}

function displaySuggestions(suggestions, metadata) {
    const container = document.getElementById('suggestions-list');
    
    if (!suggestions || suggestions.length === 0) {
        container.innerHTML = '<div class="empty-state">No suggestions found. Try a different search term.</div>';
        return;
    }

    let html = '<div class="suggestions-title">Suggestions</div>';
    
    suggestions.forEach(suggestion => {
        html += `
            <div class="suggestion-item">
                <span class="suggestion-text">${suggestion.text}</span>
                <span class="suggestion-score">${suggestion.score}</span>
            </div>
        `;
    });

    container.innerHTML = html;
}

function displaySearchResults(results, query) {
    const container = document.getElementById('products-container');
    const resultsQuery = document.getElementById('results-query');
    const resultsCount = document.getElementById('results-count');
    const noResults = document.getElementById('no-results');

    resultsQuery.textContent = `for "${query}"`;
    resultsCount.textContent = `${results.length} ${results.length === 1 ? 'result' : 'results'}`;

    if (!results || results.length === 0) {
        showNoResults();
        return;
    }

    noResults.style.display = 'none';
    
    const validResults = results.filter(product => 
        product.product_id !== 'fallback_1' && 
        product.product_id !== 'search_error' && 
        product.title !== 'Search system not loaded'
    );

    if (validResults.length === 0) {
        showNoResults();
        return;
    }

    if (currentView === 'grid') {
        container.className = 'products-grid';
        container.innerHTML = validResults.map(product => renderProductCard(product)).join('');
    } else {
        container.className = 'products-list';
        container.innerHTML = validResults.map(product => renderProductListItem(product)).join('');
    }
}

function renderProductCard(product) {
    const price = formatPrice(product.price);
    const rating = formatRating(product.rating);
    
    return `
        <div class="product-card">
            <div class="product-title">${escapeHtml(product.title)}</div>
            <div class="product-brand">${escapeHtml(product.brand || 'Unknown Brand')}</div>
            <div class="product-category">${escapeHtml(product.category || 'General')}</div>
            <div class="product-price">${price}</div>
            <div class="product-rating">⭐ ${rating}</div>
        </div>
    `;
}

function renderProductListItem(product) {
    const price = formatPrice(product.price);
    const rating = formatRating(product.rating);
    
    return `
        <div class="product-card product-list-item">
            <div class="product-list-info">
                <div class="product-title">${escapeHtml(product.title)}</div>
                <div class="product-brand">${escapeHtml(product.brand || 'Unknown Brand')}</div>
                <div class="product-category">${escapeHtml(product.category || 'General')}</div>
                <div class="product-rating">⭐ ${rating}</div>
            </div>
            <div class="product-list-price">${price}</div>
        </div>
    `;
}

function switchView(view) {
    currentView = view;
    
    const gridBtn = document.getElementById('grid-view');
    const listBtn = document.getElementById('list-view');
    
    if (view === 'grid') {
        gridBtn.classList.add('active');
        listBtn.classList.remove('active');
    } else {
        listBtn.classList.add('active');
        gridBtn.classList.remove('active');
    }

    const container = document.getElementById('products-container');
    if (container.children.length > 0 && !container.querySelector('.loading')) {
        const query = document.getElementById('search-input').value.trim();
        if (query) {
            performSearch(query);
        }
    }
}

function showSearchResults() {
    document.getElementById('search-results').classList.add('show');
}

function hideSearchResults() {
    document.getElementById('search-results').classList.remove('show');
}

function hideSuggestions() {
    document.getElementById('suggestions-container').style.display = 'none';
}

function showSuggestions() {
    document.getElementById('suggestions-container').style.display = 'block';
}

function showEmptyState() {
    document.getElementById('suggestions-list').innerHTML = 
        '<div class="empty-state">Start typing to see intelligent suggestions</div>';
    showSuggestions();
}

function showError(message) {
    document.getElementById('suggestions-list').innerHTML = 
        `<div class="empty-state">Error: ${message}</div>`;
}

function showNoResults() {
    document.getElementById('products-container').innerHTML = '';
    document.getElementById('no-results').style.display = 'block';
}

function updateStatus(message) {
    document.getElementById('status-bar').textContent = message;
}

function formatPrice(price) {
    if (!price || price === 0) return 'Price not available';
    return `₹${price.toLocaleString('en-IN')}`;
}

function formatRating(rating) {
    if (!rating || rating === 0) return 'No rating';
    return rating.toFixed(1);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', init);