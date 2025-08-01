// Enhanced Frontend Script with Auto-Refresh and Shared Data
// ==========================================================

let config = {};
let currentQuery = '';
let debounceTimer = null;
let currentView = 'grid';

// Enhanced search state management
let searchState = {
    currentQuery: '',
    currentResults: [],
    isSearchActive: false,
    lastPersona: '',
    lastLocation: '',
    lastEvent: ''
};

// Initialize the application
async function init() {
    try {
        // Load configuration
        const response = await fetch('/api/config');
        config = await response.json();
        
        // Populate dropdowns
        populatePersonas();
        populateLocations();
        populateEvents();
        
        // Set up event listeners
        setupEventListeners();
        
        // Initialize with first persona
        const firstPersona = config.personas[0]?.id || 'tech_enthusiast';
        updatePersonaInfo(firstPersona);
        
        // Store initial state
        searchState.lastPersona = firstPersona;
        searchState.lastLocation = config.locations[0]?.id || 'Mumbai';
        searchState.lastEvent = config.events[0]?.id || 'none';
        
        showEmptyState();
        
    } catch (error) {
        console.error('Failed to initialize:', error);
        showError('Failed to load configuration');
    }
}

function populatePersonas() {
    const select = document.getElementById('persona-select');
    select.innerHTML = config.personas.map(persona => 
        `<option value="${persona.id}">${persona.name}</option>`
    ).join('');
}

function populateLocations() {
    const select = document.getElementById('location-select');
    select.innerHTML = config.locations.map(location => 
        `<option value="${location.id}">${location.name}</option>`
    ).join('');
}

function populateEvents() {
    const select = document.getElementById('event-select');
    select.innerHTML = config.events.map(event => 
        `<option value="${event.id}">${event.name}</option>`
    ).join('');
}

function updatePersonaInfo(personaId) {
    const personaInfoDiv = document.getElementById('persona-info');
    if (!personaInfoDiv) return;
    
    const persona = config.personas.find(p => p.id === personaId);
    if (persona && persona.description) {
        personaInfoDiv.innerHTML = `
            <strong>${persona.name}</strong><br>
            <span style="color: #7f8c8d; font-size: 11px;">${persona.description}</span>
        `;
    } else {
        personaInfoDiv.innerHTML = 'Select a persona to see personalized suggestions';
    }
    
    console.log('🎭 Persona updated to:', personaId, persona);
}

function setupEventListeners() {
    const searchInput = document.getElementById('search-input');
    const personaSelect = document.getElementById('persona-select');
    const locationSelect = document.getElementById('location-select');
    const eventSelect = document.getElementById('event-select');

    // Search input with debouncing for autosuggest
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

    // Enhanced persona change with auto-refresh
    personaSelect.addEventListener('change', (e) => {
        const newPersona = e.target.value;
        updatePersonaInfo(newPersona);
        
        // Auto-refresh search results if search is active and persona changed
        if (searchState.isSearchActive && 
            searchState.currentQuery && 
            newPersona !== searchState.lastPersona) {
            
            console.log(`🔄 Persona changed from ${searchState.lastPersona} to ${newPersona}, refreshing search...`);
            searchState.lastPersona = newPersona;
            performSearch(searchState.currentQuery);
        } else {
            searchState.lastPersona = newPersona;
        }
    });

    // Location change with auto-refresh
    locationSelect.addEventListener('change', (e) => {
        const newLocation = e.target.value;
        
        if (searchState.isSearchActive && 
            searchState.currentQuery && 
            newLocation !== searchState.lastLocation) {
            
            console.log(`🔄 Location changed to ${newLocation}, refreshing search...`);
            searchState.lastLocation = newLocation;
            performSearch(searchState.currentQuery);
        } else {
            searchState.lastLocation = newLocation;
        }
    });

    // Event change with auto-refresh
    eventSelect.addEventListener('change', (e) => {
        const newEvent = e.target.value;
        
        if (searchState.isSearchActive && 
            searchState.currentQuery && 
            newEvent !== searchState.lastEvent) {
            
            console.log(`🔄 Event changed to ${newEvent}, refreshing search...`);
            searchState.lastEvent = newEvent;
            performSearch(searchState.currentQuery);
        } else {
            searchState.lastEvent = newEvent;
        }
    });

    // Suggestion click handlers
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('suggestion-item')) {
            const suggestion = e.target.textContent.trim();
            document.getElementById('search-input').value = suggestion;
            performSearch(suggestion);
        }
    });

    // View toggle with shared data (no refetch)
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

    // Update search state
    searchState.currentQuery = query;
    searchState.isSearchActive = true;

    hideSuggestions();
    showSearchResults();
    showSearchLoading();

    try {
        const requestData = {
            query: query,
            context: {
                location: document.getElementById('location-select').value,
                persona_tag: document.getElementById('persona-select').value,
                event: document.getElementById('event-select').value
            },
            top_k: 50  // Request 50 results as specified
        };

        const response = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        const results = await response.json();
        
        if (response.ok && Array.isArray(results)) {
            // Store results for view switching (shared data)
            searchState.currentResults = results;
            displaySearchResults(results, query);
            updateStatus(`Found ${results.length} products`);
        } else {
            showNoResults();
            updateStatus('No products found');
            searchState.currentResults = [];
        }
    } catch (error) {
        console.error('Search error:', error);
        showNoResults();
        updateStatus('Search failed');
        searchState.currentResults = [];
    }
}

function showLoading() {
    document.getElementById('loading').classList.add('show');
    document.getElementById('suggestions-list').innerHTML = '<div class="loading show">Getting suggestions...</div>';
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

    let html = '<div class="suggestions-title">Enhanced Suggestions</div>';
    html += suggestions.map((suggestion, index) => {
        const text = Array.isArray(suggestion) ? suggestion[0] : suggestion;
        const score = Array.isArray(suggestion) ? suggestion[1] : 1.0;
        return `<div class="suggestion-item" data-score="${score}">${escapeHtml(text)}</div>`;
    }).join('');
    
    container.innerHTML = html;
    showSuggestions();
}

function displaySearchResults(results, query) {
    const container = document.getElementById('products-container');
    const resultsQuery = document.getElementById('results-query');
    const resultsCount = document.getElementById('results-count');
    const noResults = document.getElementById('no-results');

    resultsQuery.textContent = `for "${query}"`;
    resultsCount.textContent = `${results.length} ${results.length === 1 ? 'result' : 'results'}`;
    
    // Update left sidebar result counter
    updateResultCounter(results.length, query);

    if (!results || results.length === 0) {
        showNoResults();
        return;
    }

    noResults.style.display = 'none';
    
    // Filter out fallback/error products
    const validResults = results.filter(product => 
        product.product_id !== 'fallback_1' && 
        product.product_id !== 'search_error' && 
        product.product_id !== 'search_unavailable' &&
        product.title !== 'Search system not loaded'
    );

    if (validResults.length === 0) {
        showNoResults();
        return;
    }

    // Use shared data - no API call needed
    if (currentView === 'grid') {
        container.className = 'products-grid';
        container.innerHTML = validResults.map(product => renderProductCard(product)).join('');
    } else {
        container.className = 'products-list';
        container.innerHTML = validResults.map(product => renderProductListItem(product)).join('');
    }
}

// Enhanced minimalistic product card (2D, no shadows)
function renderProductCard(product) {
    const price = formatPrice(product.price);
    const rating = formatRating(product.rating);
    const title = escapeHtml(product.title);
    const brand = escapeHtml(product.brand);
    const category = escapeHtml(product.category);
    const assured = product.is_f_assured ? '<span class="f-assured">F-Assured</span>' : '';
    
    // Enhanced info with persona scoring if available
    const personaInfo = product.persona_score ? `<div class="persona-score">Personalized: ${(product.persona_score * 100).toFixed(0)}%</div>` : '';
    
    return `
        <div class="product-card" data-product-id="${product.product_id}">
            <div class="product-image-container">
                <img src="${product.image_url || '/static/placeholder.jpg'}" alt="${title}" class="product-image">
                ${assured}
            </div>
            <div class="product-info">
                <h3 class="product-title">${title}</h3>
                <div class="product-brand">${brand}</div>
                <div class="product-category">${category}</div>
                <div class="product-price">${price}</div>
                <div class="product-rating">${rating}</div>
                ${personaInfo}
            </div>
        </div>
    `;
}

// Enhanced minimalistic list item (2D, no shadows)
function renderProductListItem(product) {
    const price = formatPrice(product.price);
    const rating = formatRating(product.rating);
    const title = escapeHtml(product.title);
    const brand = escapeHtml(product.brand);
    const category = escapeHtml(product.category);
    const assured = product.is_f_assured ? '<span class="f-assured">F-Assured</span>' : '';
    const description = escapeHtml(product.description || '');
    
    // Enhanced info with persona scoring if available
    const personaInfo = product.persona_score ? `<span class="persona-score">Match: ${(product.persona_score * 100).toFixed(0)}%</span>` : '';
    
    return `
        <div class="product-list-item" data-product-id="${product.product_id}">
            <div class="product-image-container">
                <img src="${product.image_url || '/static/placeholder.jpg'}" alt="${title}" class="product-image">
            </div>
            <div class="product-details">
                <h3 class="product-title">${title}</h3>
                <div class="product-brand-category">${brand} • ${category}</div>
                <div class="product-description">${description}</div>
                <div class="product-meta">
                    <span class="product-price">${price}</span>
                    <span class="product-rating">${rating}</span>
                    ${assured}
                    ${personaInfo}
                </div>
            </div>
        </div>
    `;
}

// Enhanced view switching with shared data (no refetch)
function switchView(newView) {
    if (currentView === newView) return;
    
    currentView = newView;
    
    // Update active button
    document.querySelectorAll('.view-toggle button').forEach(btn => {
        btn.classList.remove('active');
    });
    document.getElementById(`${newView}-view`).classList.add('active');
    
    // Re-render using stored data (no API call)
    if (searchState.currentResults && searchState.currentResults.length > 0) {
        displaySearchResults(searchState.currentResults, searchState.currentQuery);
        console.log(`🔄 Switched to ${newView} view using cached data`);
    }
}

// UI state management functions
function showSearchResults() {
    document.getElementById('search-results').classList.add('show');
}

function hideSearchResults() {
    document.getElementById('search-results').classList.remove('show');
    searchState.isSearchActive = false;
}

function hideSuggestions() {
    document.getElementById('suggestions-container').style.display = 'none';
}

function showSuggestions() {
    document.getElementById('suggestions-container').style.display = 'block';
}

function showEmptyState() {
    hideSuggestions();
    hideSearchResults();
    document.getElementById('loading').classList.remove('show');
}

function showError(message) {
    document.getElementById('suggestions-list').innerHTML = `<div class="error-state">⚠️ ${escapeHtml(message)}</div>`;
    showSuggestions();
}

function showNoResults() {
    document.getElementById('no-results').style.display = 'block';
    document.getElementById('products-container').innerHTML = '';
}

function updateStatus(message) {
    console.log(`Status: ${message}`);
}

// Utility functions
function formatPrice(price) {
    if (!price || price === 0) return 'Price not available';
    return `₹${parseFloat(price).toLocaleString('en-IN')}`;
}

function formatRating(rating) {
    if (!rating || rating === 0) return 'No rating';
    return `⭐ ${parseFloat(rating).toFixed(1)}`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', init);