/**
 * analyze.js
 * Handles drag-and-drop file uploads, image validation, preview rendering,
 * and AJAX prediction requests with multi-stage progress animation.
 */

document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const previewBox = document.getElementById("previewContainer");
  const previewImage = document.getElementById("previewImage");
  const fileNameDisplay = document.getElementById("fileNameDisplay");
  const fileSizeDisplay = document.getElementById("fileSizeDisplay");
  const removeBtn = document.getElementById("removeImageBtn");
  const analyzeBtn = document.getElementById("startAnalyzeBtn");
  const progressBox = document.getElementById("progressBox");
  const progressStageText = document.getElementById("progressStageText");
  const modelDisconnectedAlert = document.getElementById("modelDisconnectedAlert");

  let selectedFile = null;

  if (!dropzone || !fileInput) return;

  // Click dropzone to open file dialog
  dropzone.addEventListener("click", () => fileInput.click());

  // Drag & Drop event handlers
  ["dragenter", "dragover"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("drag-over");
    });
  });

  // Handle file drop
  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelection(files[0]);
    }
  });

  // Handle file input change
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      handleFileSelection(fileInput.files[0]);
    }
  });

  // Validate and display file preview
  function handleFileSelection(file) {
    // Reset previous alerts
    if (modelDisconnectedAlert) modelDisconnectedAlert.style.display = "none";

    const allowedTypes = ["image/jpeg", "image/png", "image/webp", "image/jpg"];
    if (!allowedTypes.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|webp)$/i)) {
      showToast("Invalid file format. Please select a JPG, PNG, or WEBP image.", "error");
      return;
    }

    const maxSizeBytes = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSizeBytes) {
      showToast("Image size is too large. Maximum allowed size is 10MB.", "error");
      return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      fileNameDisplay.textContent = file.name;
      fileSizeDisplay.textContent = formatBytes(file.size);
      
      dropzone.style.display = "none";
      previewBox.style.display = "block";
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // Remove/change image button
  if (removeBtn) {
    removeBtn.addEventListener("click", () => {
      selectedFile = null;
      fileInput.value = "";
      previewBox.style.display = "none";
      dropzone.style.display = "block";
      if (progressBox) progressBox.style.display = "none";
    });
  }

  // Analyze button submit
  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", async () => {
      if (!selectedFile) {
        showToast("Please upload an image first.", "error");
        return;
      }

      // Start loading animation
      previewBox.style.display = "none";
      progressBox.style.display = "block";
      analyzeBtn.disabled = true;

      // Multi-stage simulated progress milestones
      const stages = [
        "1/4: Initializing OpenCV & validating image structure...",
        "2/4: Resizing lesion image to 224x224 RGB & normalizing pixel array...",
        "3/4: Passing tensor through Deep CNN / Transfer Learning layers...",
        "4/4: Calculating Softmax probabilities across 7 HAM10000 classes..."
      ];

      let stageIdx = 0;
      progressStageText.textContent = stages[0];
      const stageInterval = setInterval(() => {
        stageIdx = (stageIdx + 1) % stages.length;
        progressStageText.textContent = stages[stageIdx];
        updateStepIndicators(stageIdx);
      }, 700);

      const formData = new FormData();
      formData.append("image", selectedFile);

      try {
        const response = await fetch("/api/predict", {
          method: "POST",
          body: formData
        });

        clearInterval(stageInterval);

        const data = await response.json();

        if (response.status === 503 || (data.model_connected === false)) {
          // Model disconnected state
          progressBox.style.display = "none";
          previewBox.style.display = "block";
          analyzeBtn.disabled = false;

          if (modelDisconnectedAlert) {
            modelDisconnectedAlert.style.display = "block";
            modelDisconnectedAlert.scrollIntoView({ behavior: "smooth" });
          } else {
            showToast("AI Model is not connected. Please check /model-status.", "error", 6000);
          }
          return;
        }

        if (data.success && data.result_url) {
          // Success: redirect to result page
          progressStageText.textContent = "Screening complete! Redirecting to report...";
          window.location.href = data.result_url;
        } else {
          progressBox.style.display = "none";
          previewBox.style.display = "block";
          analyzeBtn.disabled = false;
          showToast(data.error || "Failed to analyze image.", "error", 5000);
        }

      } catch (err) {
        clearInterval(stageInterval);
        progressBox.style.display = "none";
        previewBox.style.display = "block";
        analyzeBtn.disabled = false;
        showToast("Network error while connecting to Flask API: " + err.message, "error");
      }
    });
  }

  function updateStepIndicators(idx) {
    const steps = document.querySelectorAll(".progress-step-item");
    steps.forEach((step, i) => {
      if (i <= idx) {
        step.classList.add("active");
      } else {
        step.classList.remove("active");
      }
    });
  }

  function formatBytes(bytes, decimals = 1) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
  }
});
