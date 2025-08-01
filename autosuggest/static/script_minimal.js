// script_minimal.js
document.addEventListener('DOMContentLoaded', () => {
    const app = new AutosuggestMinimalApp();
    app.init();
});

class AutosuggestMinimalApp {
    constructor() {
        this.config = null;
        this.currentSettings = {};
        this.debounceTimer = null;
        this.activeSuggestionIndex = -1;

        // DOM Elements
        this.searchInput = document.getElementById('search-input');
        this.suggestionsList = document.getElementById('suggestions-list');
        this.personaSelect = document.getElementById('persona-select');
        this.locationSelect = document.getElementById('location-select');
        this.eventSelect = document.getElementById('event-select');
    }

    async init() {
        await this.loadConfig();
        this.populateSelects();
        this.bindEvents();
    }

    async loadConfig() {
        try {
            const response = await fetch('/api/config');
            this.config = await response.json();
            this.currentSettings = {
                persona: Object.keys(this.config.personas)[0],
                location: this.config.locations[0],
                event: this.config.events[0].id,
            };
        } catch (e) {
            console.error("Failed to load config", e);
        }
    }

    populateSelects() {
        // Personas
        Object.entries(this.config.personas).forEach(([id, persona]) => {
            const option = new Option(`${persona.name}`, id);
            this.personaSelect.add(option);
        });

        // Locations
        this.config.locations.forEach(location => {
            const option = new Option(location, location);
            this.locationSelect.add(option);
        });

        // Events
        this.config.events.forEach(event => {
            const option = new Option(event.name, event.id);
            this.eventSelect.add(option);
        });
    }

    bindEvents() {
        this.searchInput.addEventListener('input', () => this.onInputChange());
        this.searchInput.addEventListener('keydown', (e) => this.onKeyDown(e));

        [this.personaSelect, this.locationSelect, this.eventSelect].forEach(select => {
            select.addEventListener('change', (e) => this.onSettingsChange(e));
        });
        
        // Clicks outside the search area
        document.addEventListener('click', (e) => {
            if (!this.searchInput.contains(e.target)) {
                this.suggestionsList.innerHTML = '';
            }
        });
    }

    onInputChange() {
        clearTimeout(this.debounceTimer);
        this.debounceTimer = setTimeout(() => {
            const query = this.searchInput.value.trim();
            if (query) {
                this.fetchSuggestions(query);
            } else {
                this.suggestionsList.innerHTML = '<div class="placeholder"><p>Suggestions will appear here as you type.</p></div>';
            }
        }, 150); // Debounce time
    }

    onSettingsChange(e) {
        this.currentSettings[e.target.id.split('-')[0]] = e.target.value;
        const query = this.searchInput.value.trim();
        if (query) {
            this.fetchSuggestions(query);
        }
    }

    onKeyDown(e) {
        const items = this.suggestionsList.querySelectorAll('.suggestion-item');
        if (!items.length) return;

        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                this.activeSuggestionIndex = (this.activeSuggestionIndex + 1) % items.length;
                this.updateActiveSuggestion(items);
                break;
            case 'ArrowUp':
                e.preventDefault();
                this.activeSuggestionIndex = (this.activeSuggestionIndex - 1 + items.length) % items.length;
                this.updateActiveSuggestion(items);
                break;
            case 'Enter':
                e.preventDefault();
                if (this.activeSuggestionIndex > -1) {
                    items[this.activeSuggestionIndex].click();
                }
                break;
            case 'Escape':
                this.suggestionsList.innerHTML = '';
                break;
        }
    }
    
    updateActiveSuggestion(items) {
        items.forEach(item => item.classList.remove('active'));
        if (this.activeSuggestionIndex > -1) {
            items[this.activeSuggestionIndex].classList.add('active');
            this.searchInput.value = items[this.activeSuggestionIndex].dataset.text;
        }
    }

    async fetchSuggestions(query) {
        const payload = { query, ...this.currentSettings };
        try {
            const response = await fetch('/api/suggest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const { suggestions } = await response.json();
            this.renderSuggestions(suggestions, query);
        } catch (e) {
            console.error("Failed to fetch suggestions", e);
            this.suggestionsList.innerHTML = '<div class="no-results">Could not fetch suggestions.</div>';
        }
    }

    renderSuggestions(suggestions, query) {
        this.activeSuggestionIndex = -1;
        if (!suggestions || suggestions.length === 0) {
            this.suggestionsList.innerHTML = '<div class="no-results">No suggestions found.</div>';
            return;
        }

        const queryRegex = new RegExp(`(${this.escapeRegex(query)})`, 'gi');
        this.suggestionsList.innerHTML = suggestions.map(s => {
            const highlightedText = s.text.replace(queryRegex, '<strong>$1</strong>');
            return `<div class="suggestion-item" data-text="${s.text}">${highlightedText}</div>`;
        }).join('');

        this.suggestionsList.querySelectorAll('.suggestion-item').forEach(item => {
            item.addEventListener('click', () => {
                this.searchInput.value = item.dataset.text;
                this.suggestionsList.innerHTML = '';
            });
        });
    }

    escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
}