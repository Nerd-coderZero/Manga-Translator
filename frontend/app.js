// Shared API client for index.html and reader.html.
//
// API_BASE is empty by default because the Space serves the frontend from
// the same origin as the API. It falls back to a query parameter so the
// same files can be opened against a backend on another origin during
// development without editing them.
const API_BASE = (function () {
  const fromQuery = new URLSearchParams(window.location.search).get("api");
  if (fromQuery) return fromQuery.replace(/\/$/, "");
  return "";
})();

function apiUrl(path) {
  return API_BASE + path;
}

// Image URLs come back from the API as same-origin absolute paths. When the
// frontend is running against a different origin they need the base applied,
// otherwise the browser resolves them against the page's own origin.
function resolveImageUrl(url) {
  if (!url) return "";
  if (/^https?:\/\//i.test(url)) return url;
  return apiUrl(url);
}

async function apiFetch(path, options) {
  const response = await fetch(apiUrl(path), options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (body && body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (err) {
      // non-JSON error body, keep the status text
    }
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return response;
}

async function getHealth() {
  const response = await apiFetch("/api/health");
  return response.json();
}

async function submitBatch(files, sourceLang) {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  form.append("source_lang", sourceLang);
  const response = await apiFetch("/api/batch", { method: "POST", body: form });
  return response.json();
}

async function getBatch(batchId) {
  const response = await apiFetch("/api/batch/" + encodeURIComponent(batchId));
  return response.json();
}

async function getBatchPage(batchId, pageIndex) {
  const response = await apiFetch(
    "/api/batch/" + encodeURIComponent(batchId) + "/page/" + pageIndex
  );
  return response.json();
}

function batchDownloadUrl(batchId) {
  return apiUrl("/api/batch/" + encodeURIComponent(batchId) + "/download");
}

async function deleteBatch(batchId) {
  const response = await apiFetch("/api/batch/" + encodeURIComponent(batchId), {
    method: "DELETE",
  });
  return response.json();
}

function readerUrl(batchId) {
  const params = new URLSearchParams({ batch: batchId });
  const api = new URLSearchParams(window.location.search).get("api");
  if (api) params.set("api", api);
  return "reader.html?" + params.toString();
}
