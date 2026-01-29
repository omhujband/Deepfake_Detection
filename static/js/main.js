document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadForm = document.getElementById('upload-form');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loading = document.getElementById('loading');
    const resultsCard = document.getElementById('results-card');
    const dropZoneContent = document.getElementById('drop-zone-content');
    const filePreview = document.getElementById('file-preview');
    const imagePreview = document.getElementById('image-preview');
    const videoPreview = document.getElementById('video-preview');
    const fileName = document.getElementById('file-name');
    const removeFileBtn = document.getElementById('remove-file');

    let selectedFile = null;
    let frameChart = null;

    // Drop zone click
    dropZone.addEventListener('click', (e) => {
        if (e.target === removeFileBtn || removeFileBtn.contains(e.target)) return;
        fileInput.click();
    });

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Remove file
    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetForm();
    });

    function handleFile(file) {
        const validExtensions = /\.(png|jpg|jpeg|webp|mp4|avi|mov|mkv)$/i;
        
        if (!validExtensions.test(file.name)) {
            alert('Invalid file type. Please upload an image or video.');
            return;
        }

        if (file.size > 100 * 1024 * 1024) {
            alert('File is too large. Maximum size is 100MB.');
            return;
        }

        selectedFile = file;
        fileName.textContent = file.name;
        dropZoneContent.classList.add('d-none');
        filePreview.classList.remove('d-none');

        // Show preview
        const isImage = /\.(png|jpg|jpeg|webp)$/i.test(file.name);
        if (isImage) {
            imagePreview.classList.remove('d-none');
            videoPreview.classList.add('d-none');
            imagePreview.src = URL.createObjectURL(file);
        } else {
            videoPreview.classList.remove('d-none');
            imagePreview.classList.add('d-none');
            videoPreview.src = URL.createObjectURL(file);
        }

        analyzeBtn.disabled = false;
    }

    function resetForm() {
        selectedFile = null;
        fileInput.value = '';
        dropZoneContent.classList.remove('d-none');
        filePreview.classList.add('d-none');
        imagePreview.classList.add('d-none');
        videoPreview.classList.add('d-none');
        analyzeBtn.disabled = true;
        resultsCard.classList.add('d-none');
    }

    // Form submit
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!selectedFile) return;

        uploadForm.classList.add('d-none');
        loading.classList.remove('d-none');
        resultsCard.classList.add('d-none');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                displayResults(result);
            } else {
                alert(result.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during analysis');
        } finally {
            loading.classList.add('d-none');
            uploadForm.classList.remove('d-none');
        }
    });

    function displayResults(result) {
        resultsCard.classList.remove('d-none');

        const verdictIcon = document.getElementById('verdict-icon');
        const verdictText = document.getElementById('verdict-text');
        const confidenceText = document.getElementById('confidence-text');
        const resultsHeader = document.getElementById('results-header');
        const realBar = document.getElementById('real-bar');
        const fakeBar = document.getElementById('fake-bar');
        const detailsList = document.getElementById('details-list');
        const frameChartContainer = document.getElementById('frame-chart-container');

        // Set verdict
        if (result.prediction === 'Fake') {
            verdictIcon.innerHTML = '<i class="fas fa-exclamation-triangle text-danger"></i>';
            verdictText.textContent = 'LIKELY FAKE';
            verdictText.className = 'fw-bold text-danger';
            resultsHeader.className = 'card-header py-3 bg-danger text-white';
        } else {
            verdictIcon.innerHTML = '<i class="fas fa-check-circle text-success"></i>';
            verdictText.textContent = 'LIKELY REAL';
            verdictText.className = 'fw-bold text-success';
            resultsHeader.className = 'card-header py-3 bg-success text-white';
        }

        confidenceText.textContent = `Confidence: ${result.confidence}%`;

        // Update bars
        realBar.style.width = `${result.real_probability}%`;
        realBar.textContent = `${result.real_probability}%`;
        fakeBar.style.width = `${result.fake_probability}%`;
        fakeBar.textContent = `${result.fake_probability}%`;

        // Details
        let detailsHTML = `
            <li><strong>File:</strong> ${result.filename}</li>
            <li><strong>Type:</strong> ${result.file_type}</li>
            <li><strong>Processing Time:</strong> ${result.processing_time}s</li>
            <li><strong>Model Loaded:</strong> ${result.model_loaded ? 'Yes' : 'No (using random weights)'}</li>
        `;

        if (result.face_detected !== undefined) {
            detailsHTML += `<li><strong>Face Detected:</strong> ${result.face_detected ? 'Yes' : 'No'}</li>`;
        }

        if (result.file_type === 'video') {
            detailsHTML += `
                <li><strong>Frames Analyzed:</strong> ${result.frames_analyzed}</li>
                <li><strong>Fake Frame Ratio:</strong> ${result.fake_frame_ratio}%</li>
            `;
            if (result.metadata) {
                detailsHTML += `
                    <li><strong>Duration:</strong> ${result.metadata.duration.toFixed(1)}s</li>
                    <li><strong>FPS:</strong> ${result.metadata.fps.toFixed(1)}</li>
                `;
            }

            // Frame chart
            if (result.frame_predictions && result.frame_predictions.length > 0) {
                frameChartContainer.classList.remove('d-none');
                createFrameChart(result.frame_predictions);
            }
        } else {
            frameChartContainer.classList.add('d-none');
        }

        detailsList.innerHTML = detailsHTML;
        resultsCard.scrollIntoView({ behavior: 'smooth' });
    }

    function createFrameChart(predictions) {
        const ctx = document.getElementById('frame-chart').getContext('2d');
        
        if (frameChart) frameChart.destroy();

        frameChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: predictions.map((_, i) => `Frame ${i + 1}`),
                datasets: [{
                    label: 'Fake Probability (%)',
                    data: predictions.map(p => (p * 100).toFixed(1)),
                    borderColor: 'rgb(220, 53, 69)',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { beginAtZero: true, max: 100 }
                }
            }
        });
    }
});