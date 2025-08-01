// Additional JavaScript for Result Counter and Pagination
// Add this to the existing script_enhanced.js

// Enhanced displaySearchResults with counter and pagination
function displaySearchResultsEnhanced(results, query) {
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

    // Store results for pagination
    searchState.currentResults = validResults;
    
    // Show first page (10 results)
    showResultsPage(1, validResults, query);
    
    // Setup pagination if needed
    setupPagination(validResults.length, query);

    showSearchResults();
}

function showResultsPage(pageNumber, allResults, query) {
    const RESULTS_PER_PAGE = 10;
    const startIndex = (pageNumber - 1) * RESULTS_PER_PAGE;
    const endIndex = startIndex + RESULTS_PER_PAGE;
    
    const pageResults = allResults.slice(startIndex, endIndex);
    const container = document.getElementById('products-container');
    
    // Use shared data - no API call needed
    if (currentView === 'grid') {
        container.className = 'products-grid';
        container.innerHTML = pageResults.map(product => renderProductCard(product)).join('');
    } else {
        container.className = 'products-list';
        container.innerHTML = pageResults.map(product => renderProductListItem(product)).join('');
    }
}

function updateResultCounter(count, query) {
    // Add result counter to left sidebar
    const sidebar = document.querySelector('.search-settings');
    let counterDiv = document.getElementById('results-counter');
    
    if (!counterDiv) {
        counterDiv = document.createElement('div');
        counterDiv.id = 'results-counter';
        counterDiv.className = 'results-counter';
        sidebar.appendChild(counterDiv);
    }
    
    counterDiv.innerHTML = `
        <h3>Search Results</h3>
        <div class="counter-info">
            <span class="result-count">${count}</span> 
            <span class="result-text">${count === 1 ? 'result' : 'results'}</span>
            <span class="result-query">for "${query}"</span>
        </div>
    `;
}

function setupPagination(totalResults, query) {
    const RESULTS_PER_PAGE = 10;
    const totalPages = Math.ceil(totalResults / RESULTS_PER_PAGE);
    
    let paginationDiv = document.getElementById('pagination-controls');
    if (paginationDiv) {
        paginationDiv.remove(); // Remove existing pagination
    }
    
    if (totalPages <= 1) return; // No pagination needed
    
    paginationDiv = document.createElement('div');
    paginationDiv.id = 'pagination-controls';
    paginationDiv.className = 'pagination-controls';
    
    const searchSection = document.getElementById('search-results-section');
    searchSection.appendChild(paginationDiv);
    
    // Create pagination buttons
    let paginationHTML = '<div class="pagination-info">Page 1 of ' + totalPages + '</div>';
    paginationHTML += '<div class="pagination-buttons">';
    
    for (let i = 1; i <= Math.min(totalPages, 5); i++) {
        paginationHTML += `<button class="page-btn ${i === 1 ? 'active' : ''}" onclick="goToPage(${i})">${i}</button>`;
    }
    
    if (totalPages > 5) {
        paginationHTML += '<span>...</span>';
        paginationHTML += `<button class="page-btn" onclick="goToPage(${totalPages})">${totalPages}</button>`;
    }
    
    paginationHTML += '</div>';
    paginationDiv.innerHTML = paginationHTML;
}

function goToPage(pageNumber) {
    if (!searchState.currentResults) return;
    
    const RESULTS_PER_PAGE = 10;
    const totalPages = Math.ceil(searchState.currentResults.length / RESULTS_PER_PAGE);
    
    // Show the requested page
    showResultsPage(pageNumber, searchState.currentResults);
    
    // Update pagination buttons
    document.querySelectorAll('.page-btn').forEach(btn => btn.classList.remove('active'));
    const activeBtn = document.querySelector(`[onclick="goToPage(${pageNumber})"]`);
    if (activeBtn) activeBtn.classList.add('active');
    
    // Update pagination info
    const paginationInfo = document.querySelector('.pagination-info');
    if (paginationInfo) {
        paginationInfo.textContent = `Page ${pageNumber} of ${totalPages}`;
    }
    
    // Scroll to top of results
    document.getElementById('search-results-section').scrollIntoView({ behavior: 'smooth' });
}