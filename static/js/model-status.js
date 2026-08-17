/**
 * model-status.js
 * Admin & Model Health Dashboard interactive diagnostics and reload triggers.
 */

document.addEventListener("DOMContentLoaded", () => {
  const reloadBtn = document.getElementById("reloadModelBtn");
  const pingBtn = document.getElementById("pingApiBtn");
  const pingResultText = document.getElementById("pingResultText");

  // Reload Model Button
  if (reloadBtn) {
    reloadBtn.addEventListener("click", async () => {
      reloadBtn.disabled = true;
      reloadBtn.innerHTML = `
        <svg class="spinner-ring" style="width: 16px; height: 16px; margin: 0; border-width: 2px;" viewBox="0 0 24 24"></svg>
        Reconnecting Model...
      `;

      try {
        const res = await fetch("/api/reload-model", { method: "POST" });
        const data = await res.json();

        if (data.success && data.model_status.connected) {
          showToast("AI Model successfully connected!", "success");
          setTimeout(() => window.location.reload(), 800);
        } else {
          showToast("Model not detected: " + (data.model_status ? data.model_status.message : "Error"), "error", 6000);
          reloadBtn.disabled = false;
          reloadBtn.textContent = "Retry Connection";
        }
      } catch (err) {
        showToast("Error connecting to server: " + err.message, "error");
        reloadBtn.disabled = false;
        reloadBtn.textContent = "Retry Connection";
      }
    });
  }

  // Ping API Button
  if (pingBtn && pingResultText) {
    pingBtn.addEventListener("click", async () => {
      const startTime = performance.now();
      pingResultText.textContent = "Testing latency...";

      try {
        const res = await fetch("/api/model-status");
        const endTime = performance.now();
        const latency = Math.round(endTime - startTime);

        if (res.ok) {
          pingResultText.innerHTML = `<span style="color: #10b981; font-weight: 700;">● Online (${latency} ms)</span>`;
          showToast(`API ping successful: ${latency} ms latency`, "success");
        } else {
          pingResultText.innerHTML = `<span style="color: #ef4444; font-weight: 700;">● HTTP Error ${res.status}</span>`;
        }
      } catch (err) {
        pingResultText.innerHTML = `<span style="color: #ef4444; font-weight: 700;">● Unreachable</span>`;
        showToast("Ping failed: " + err.message, "error");
      }
    });
  }
});
