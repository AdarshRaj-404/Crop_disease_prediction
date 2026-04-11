document.addEventListener('DOMContentLoaded', () => {
    // Dropzone Elements
    const dropZone = document.getElementById('drop-zone');
    const cropSelect = document.getElementById('crop-select');
    const customCropWrapper = document.getElementById('custom-crop-wrapper');
    const customCropTrigger = document.getElementById('custom-crop-trigger');
    const customCropText = document.getElementById('custom-crop-text');
    const customCropOptions = document.getElementById('custom-crop-options');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const uploadArea = document.querySelector('.upload-area');
    
    // File Info Elements
    const fileInfoBar = document.getElementById('file-info-bar');
    const fileName = document.getElementById('fileName'); // Fixed casing issue below
    const fileSize = document.getElementById('file-size');
    const clearFileBtn = document.getElementById('clear-file-btn');
    
    // Action Elements
    const analyzeBtn = document.getElementById('analyze-btn');
    const loader = document.getElementById('analyze-loader');
    const errorMessage = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');
    
    // Main View Swaps
    const uploadWidget = document.getElementById('upload-widget');
    const resultsWidget = document.getElementById('results-widget');
    const resetWorkflowBtn = document.getElementById('reset-workflow');
    
    // Result Data Elements
    const resultImg = document.getElementById('result-img');
    const diagnosisBadge = document.getElementById('diagnosis-badge');
    const badgeText = document.getElementById('badge-text');
    const diagnosisTitle = document.getElementById('diagnosis-title');
    const diagnosisSubtitle = document.getElementById('diagnosis-subtitle');
    const confBar = document.getElementById('conf-bar');
    const confValText = document.getElementById('conf-val-text');

    // Tab Elements
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');

    let currentFile = null;

    // --- Fetch Crops ---
    fetch('/crops')
        .then(res => res.json())
        .then(data => {
            customCropText.textContent = 'Select a crop...';
            customCropOptions.innerHTML = '';
            data.crops.forEach(crop => {
                const opt = document.createElement('div');
                opt.className = 'custom-option';
                opt.textContent = crop;
                
                opt.addEventListener('click', () => {
                    cropSelect.value = crop;
                    customCropText.textContent = crop;
                    closeDropdown();
                });
                
                customCropOptions.appendChild(opt);
            });
        })
        .catch(err => console.error("Error fetching crops:", err));

    function toggleDropdown() {
        customCropOptions.classList.toggle('hidden');
        customCropTrigger.classList.toggle('active');
    }

    function closeDropdown() {
        customCropOptions.classList.add('hidden');
        customCropTrigger.classList.remove('active');
    }

    if (customCropTrigger) {
        customCropTrigger.addEventListener('click', toggleDropdown);
    }

    document.addEventListener('click', (e) => {
        if (customCropWrapper && !customCropWrapper.contains(e.target)) {
            closeDropdown();
        }
    });

    // --- Drag & Drop Handlers ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.add('highlight'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('highlight'), false);
    });

    uploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        let dt = e.dataTransfer;
        let files = dt.files;
        handleFiles(files);
    }

    // --- Click Upload Handlers ---
    browseBtn.addEventListener('click', (e) => {
        e.preventDefault(); // Prevent accidental form submissions if any
        fileInput.click();
    });
    
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function formatBytes(bytes, decimals = 2) {
        if (!+bytes) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
    }

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                currentFile = file;
                
                // Show file info
                document.getElementById('file-name').textContent = file.name;
                fileSize.textContent = formatBytes(file.size);
                
                uploadArea.classList.add('hidden');
                fileInfoBar.classList.remove('hidden');
                
                analyzeBtn.disabled = false;
                
                // Pre-load image for results view
                const reader = new FileReader();
                reader.onload = (e) => {
                    resultImg.src = e.target.result;
                };
                reader.readAsDataURL(file);
                
            } else {
                errorMessage.classList.remove('hidden');
                errorText.textContent = "Please select a valid image file.";
            }
        }
    }

    // --- Clear File ---
    clearFileBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        fileInfoBar.classList.add('hidden');
        uploadArea.classList.remove('hidden');
        analyzeBtn.disabled = true;
        errorMessage.classList.add('hidden');
    });

    // --- Reset Entire Workflow ---
    resetWorkflowBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        
        // Reset Body State
        document.body.classList.remove('results-mode');
        
        // Reset Widgets
        resultsWidget.classList.add('hidden');
        uploadWidget.classList.remove('hidden');
        
        // Reset Upload Area
        fileInfoBar.classList.add('hidden');
        uploadArea.classList.remove('hidden');
        analyzeBtn.disabled = true;
        errorMessage.classList.add('hidden');
        
        // Reset Results data
        confBar.style.width = '0%';
        confValText.textContent = '0%';
    });

    // --- Analyze Action ---
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;
        
        const selectedCrop = cropSelect.value;
        if (!selectedCrop) {
            errorMessage.classList.remove('hidden');
            errorText.textContent = "Please select a crop from the dropdown above.";
            return;
        }

        // UI Transition to Loading state within the same widget
        fileInfoBar.classList.add('hidden');
        analyzeBtn.classList.add('hidden');
        loader.classList.remove('hidden');
        errorMessage.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('crop', selectedCrop);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Error predicting");
            }

            const data = await response.json();
            displayResults(data);
            
        } catch (error) {
            console.error("Prediction Error:", error);
            
            errorMessage.classList.remove('hidden');
            errorText.textContent = `Please upload a valid ${selectedCrop} image.`;
            
            // Revert UI on error
            loader.classList.add('hidden');
            fileInfoBar.classList.remove('hidden');
            analyzeBtn.classList.remove('hidden');
        }
    });

    function displayResults(data) {
        // UI Transition to Full Width Results Mode
        document.body.classList.add('results-mode');
        uploadWidget.classList.add('hidden');
        
        // Reset loader back for next time
        loader.classList.add('hidden');
        analyzeBtn.classList.remove('hidden');
        
        // Show Results
        resultsWidget.classList.remove('hidden');

        // Populate Data
        diagnosisTitle.textContent = data.predicted_class;
        
        let isHealthy = data.raw_class.toLowerCase().includes('healthy');
        
        // Update Learn More Link
        const learnMoreLink = document.getElementById('learn-more-link');
        if (learnMoreLink) {
            const query = isHealthy ? `${data.predicted_class} cultivation best practices agriculture` : `${data.predicted_class} disease agriculture pathology`;
            learnMoreLink.href = `https://www.google.com/search?q=${encodeURIComponent(query)}`;
        }
        
        if (isHealthy) {
            diagnosisBadge.className = 'pill-badge healthy';
            badgeText.textContent = 'HEALTHY';
            diagnosisSubtitle.textContent = "No disease detected";
            confBar.className = 'conf-bar-fill'; // Default green
        } else {
            diagnosisBadge.className = 'pill-badge disease';
            badgeText.textContent = 'DISEASE DETECTED';
            diagnosisSubtitle.textContent = "Pathogen signatures identified";
            confBar.className = 'conf-bar-fill warning'; // Red
        }
        
        let confidence = data.confidence;
        confValText.textContent = Math.round(confidence) + '%';
        
        // Animate progress bar width
        setTimeout(() => {
            confBar.style.width = confidence + '%';
        }, 100);

        // Populate Dynamic Treatment List
        const treatmentList = document.getElementById('treatment-list');
        if (treatmentList && data.treatment) {
            treatmentList.innerHTML = '';
            data.treatment.forEach(step => {
                const li = document.createElement('li');
                
                const iconDiv = document.createElement('div');
                iconDiv.className = 'check-icon';
                iconDiv.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
                
                const textDiv = document.createElement('div');
                textDiv.textContent = step;
                
                li.appendChild(iconDiv);
                li.appendChild(textDiv);
                treatmentList.appendChild(li);
            });
        }
    }
    
    // --- Tabs System ---
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active from all tabs
            tabs.forEach(t => t.classList.remove('active'));
            // Remove active from all content
            tabContents.forEach(c => c.classList.remove('active'));
            
            // Add active to clicked
            tab.classList.add('active');
            
            // Show corresponding content using data attribute
            // (Only implemented Overview right now since it's hardcoded for demo, but easy to expand)
            const targetId = 'tab-' + tab.getAttribute('data-tab');
            const targetContent = document.getElementById(targetId);
            if(targetContent) {
                targetContent.classList.add('active');
            }
        });
    });
});
