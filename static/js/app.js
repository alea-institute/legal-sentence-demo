// Character counter for text input
document.addEventListener('DOMContentLoaded', function() {
    // Character counter for textarea
    const textarea = document.getElementById('text');
    const charCounter = document.getElementById('char-counter');
    
    if (textarea && charCounter) {
        // Update character count on load
        updateCharCount();
        
        // Update character count on input
        textarea.addEventListener('input', updateCharCount);
        
        function updateCharCount() {
            const count = textarea.value.length;
            charCounter.textContent = `${count.toLocaleString()} characters`;
            
            // Add warning class if approaching limit
            if (count > 40000) {
                charCounter.classList.add('text-yellow-600');
                charCounter.classList.add('font-medium');
            } else if (count > 45000) {
                charCounter.classList.remove('text-yellow-600');
                charCounter.classList.add('text-red-600');
                charCounter.classList.add('font-medium');
            } else {
                charCounter.classList.remove('text-yellow-600');
                charCounter.classList.remove('text-red-600');
                charCounter.classList.remove('font-medium');
            }
        }
    }
    
    // Setup CharBoundary threshold slider
    const cbCheckbox = document.querySelector('input[name="tokenizers"][value="charboundary"]');
    const cbSettings = document.getElementById('charboundary-settings');
    const thresholdSlider = document.getElementById('charboundary_threshold');
    const thresholdValue = document.getElementById('threshold-value');
    
    if (cbCheckbox && cbSettings && thresholdSlider && thresholdValue) {
        // Set initial visibility based on checkbox state
        cbSettings.classList.toggle('hidden', !cbCheckbox.checked);
        
        // Update the display value when slider changes
        thresholdSlider.addEventListener('input', function() {
            thresholdValue.textContent = this.value;
        });
        
        // Toggle settings visibility when checkbox changes
        cbCheckbox.addEventListener('change', function() {
            cbSettings.classList.toggle('hidden', !this.checked);
        });
    }
    
    // Load sample presets
    const presetsContainer = document.getElementById('presets-container');
    if (presetsContainer) {
        console.log('Found presets container, loading presets...');
        
        // Show debugging message
        presetsContainer.innerHTML = '<div class="text-xs text-gray-500">Loading presets...</div>';
        
        fetch('/static/js/presets.json')
            .then(response => {
                console.log('Fetch response:', response.status);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(presets => {
                console.log(`Loaded ${presets.length} presets successfully`);
                
                // Clear loading placeholder
                presetsContainer.innerHTML = '';
                
                // Add preset buttons
                presets.forEach(preset => {
                    const button = document.createElement('button');
                    button.type = 'button';
                    button.className = 'text-xs bg-pink-50 hover:bg-pink-100 text-pink-800 py-2 px-3 rounded border border-pink-200 transition-colors font-medium';
                    button.textContent = preset.name;
                    button.title = preset.description;
                    
                    // Add click handler to load preset text
                    button.addEventListener('click', () => {
                        if (textarea) {
                            textarea.value = preset.text;
                            updateCharCount();
                            
                            // If preset has threshold value for CharBoundary, set it
                            if (preset.charboundary_threshold && thresholdSlider && thresholdValue) {
                                thresholdSlider.value = preset.charboundary_threshold;
                                thresholdValue.textContent = preset.charboundary_threshold;
                            }
                        }
                    });
                    
                    presetsContainer.appendChild(button);
                });
            })
            .catch(error => {
                console.error('Error loading presets:', error);
                presetsContainer.innerHTML = `<p class="text-xs text-red-500">Error loading sample texts: ${error.message}</p>`;
            });
    }
    
    // Handle share link copy
    const shareLink = document.getElementById('share-link');
    const copyButton = document.getElementById('copy-share-link');
    
    if (shareLink && copyButton) {
        copyButton.addEventListener('click', () => {
            navigator.clipboard.writeText(shareLink.href)
                .then(() => {
                    // Show copied feedback
                    const originalText = copyButton.textContent;
                    copyButton.textContent = 'Copied!';
                    copyButton.disabled = true;
                    
                    // Reset after 2 seconds
                    setTimeout(() => {
                        copyButton.textContent = originalText;
                        copyButton.disabled = false;
                    }, 2000);
                })
                .catch(err => {
                    console.error('Failed to copy text: ', err);
                });
        });
    }
    
    // Make sentence length distribution bars animated and responsive
    // We're no longer using JavaScript animation for distribution bars
    // The bars are now directly set with proper widths in the template
    function checkDistributionBars() {
        const distributionBars = document.querySelectorAll('.distribution-bar');
        if (distributionBars.length > 0) {
            console.log(`Found ${distributionBars.length} distribution bars`);
            distributionBars.forEach(bar => {
                // Just log the current width to verify it's working
                console.log(`Bar width: ${bar.style.width}`);
            });
        } else {
            console.log('No distribution bars found - they may not be visible yet');
        }
    }
    
    // Initial check
    setTimeout(checkDistributionBars, 300);
    
    // Log when tab changes to table view
    document.addEventListener('tableViewActivated', () => {
        console.log('Table view tab activated event received');
        setTimeout(checkDistributionBars, 100);
    });
});