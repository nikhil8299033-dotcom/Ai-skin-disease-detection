/**
 * history.js
 * Manages prediction history table, real-time search, class filtering, and async record deletion.
 */

document.addEventListener("DOMContentLoaded", () => {
  const searchInput = document.getElementById("historySearch");
  const classFilter = document.getElementById("classFilter");
  const historyTableBody = document.getElementById("historyTableBody");
  const totalCountBadge = document.getElementById("historyTotalCount");
  const emptyState = document.getElementById("emptyHistoryState");

  if (!historyTableBody) return;

  // Initial load
  fetchHistory();

  // Search input with debounce
  let debounceTimeout;
  if (searchInput) {
    searchInput.addEventListener("input", () => {
      clearTimeout(debounceTimeout);
      debounceTimeout = setTimeout(() => fetchHistory(), 250);
    });
  }

  // Class filter change
  if (classFilter) {
    classFilter.addEventListener("change", () => fetchHistory());
  }

  async function fetchHistory() {
    const search = searchInput ? searchInput.value.trim() : "";
    const filter = classFilter ? classFilter.value : "all";

    let url = `/api/history?limit=100`;
    if (search) url += `&search=${encodeURIComponent(search)}`;
    if (filter && filter !== "all") url += `&class=${encodeURIComponent(filter)}`;

    try {
      const res = await fetch(url);
      const data = await res.json();

      if (data.success) {
        renderHistoryRows(data.records);
        if (totalCountBadge) totalCountBadge.textContent = `${data.records.length} records`;
      } else {
        showToast("Error loading history: " + data.error, "error");
      }
    } catch (err) {
      showToast("Failed to fetch history data.", "error");
    }
  }

  function renderHistoryRows(records) {
    historyTableBody.innerHTML = "";

    if (!records || records.length === 0) {
      if (emptyState) emptyState.style.display = "block";
      const tableWrapper = document.getElementById("historyTableWrapper");
      if (tableWrapper) tableWrapper.style.display = "none";
      return;
    }

    if (emptyState) emptyState.style.display = "none";
    const tableWrapper = document.getElementById("historyTableWrapper");
    if (tableWrapper) tableWrapper.style.display = "block";

    records.forEach(rec => {
      const tr = document.createElement("tr");
      tr.id = `history-row-${rec.id}`;

      // Badge color based on risk
      let badgeClass = "badge-blue";
      if (rec.risk_level === "Critical" || rec.risk_level === "High") {
        badgeClass = "badge-rose";
      } else if (rec.risk_level === "Moderate") {
        badgeClass = "badge-amber";
      } else {
        badgeClass = "badge-emerald";
      }

      const confPct = (rec.confidence * 100).toFixed(1) + "%";

      tr.innerHTML = `
        <td>
          <img src="${rec.image_path}" alt="Skin Lesion" class="history-thumb" onerror="this.src='/static/images/placeholder.svg'">
        </td>
        <td>
          <div style="font-weight: 700; color: var(--text-main);">${rec.prediction_name}</div>
          <div style="font-size: 0.78rem; color: var(--text-muted); text-transform: uppercase;">Code: ${rec.prediction_code}</div>
        </td>
        <td>
          <span class="badge ${badgeClass}">${rec.risk_level || 'Evaluated'}</span>
        </td>
        <td>
          <div style="font-weight: 700;">${confPct}</div>
          <div class="dist-bar-track" style="width: 80px; height: 5px; margin-top: 4px;">
            <div class="dist-bar-fill" style="width: ${confPct}; background: ${rec.confidence < 0.6 ? '#f59e0b' : '#10b981'};"></div>
          </div>
        </td>
        <td style="color: var(--text-muted); font-size: 0.85rem;">
          ${rec.created_at}
        </td>
        <td>
          <div style="display: flex; align-items: center; gap: 8px;">
            <a href="/result/${rec.id}" class="btn btn-outline btn-sm">View Report</a>
            <button class="btn-icon-delete" data-id="${rec.id}" title="Delete Record">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="3 6 5 6 21 6"></polyline>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
              </svg>
            </button>
          </div>
        </td>
      `;

      historyTableBody.appendChild(tr);
    });

    // Attach delete listeners
    document.querySelectorAll(".btn-icon-delete").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        const id = btn.getAttribute("data-id");
        if (confirm(`Are you sure you want to delete prediction record #${id}?`)) {
          await deleteRecord(id);
        }
      });
    });
  }

  async function deleteRecord(id) {
    try {
      const res = await fetch(`/api/history/${id}`, { method: "DELETE" });
      const data = await res.json();

      if (data.success) {
        showToast(`Record #${id} deleted successfully.`, "success");
        const row = document.getElementById(`history-row-${id}`);
        if (row) {
          row.style.transition = "all 0.3s ease";
          row.style.opacity = "0";
          row.style.transform = "translateX(30px)";
          setTimeout(() => {
            row.remove();
            fetchHistory(); // refresh count
          }, 300);
        }
      } else {
        showToast("Failed to delete record: " + data.error, "error");
      }
    } catch (err) {
      showToast("Error deleting record.", "error");
    }
  }
});
