// sample_codebase/web/api_client.js

const DEFAULT_RETRIES = 3;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Fetches JSON from the inventory API.
 * Failed requests and 5xx responses are retried with exponential backoff
 * (250ms, 500ms, 1000ms ...). 4xx responses are not retried.
 */
export async function fetchWithRetry(url, options = {}, retries = DEFAULT_RETRIES) {
  for (let attempt = 0; ; attempt++) {
    let response;
    try {
      response = await fetch(url, options);
    } catch (networkError) {
      if (attempt >= retries) throw networkError;
      await sleep(250 * 2 ** attempt);
      continue;
    }
    if (response.status >= 500 && attempt < retries) {
      await sleep(250 * 2 ** attempt);
      continue;
    }
    if (!response.ok) {
      throw new Error(`request failed with status ${response.status}`);
    }
    return await response.json();
  }
}

export class InventoryClient {
  constructor(baseUrl, token) {
    this.baseUrl = baseUrl;
    this.token = token;
  }

  headers() {
    return { Authorization: `Bearer ${this.token}`, "Content-Type": "application/json" };
  }

  listProducts() {
    return fetchWithRetry(`${this.baseUrl}/products`, { headers: this.headers() });
  }

  removeStock(sku, amount) {
    return fetchWithRetry(`${this.baseUrl}/products/${sku}/remove`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ amount }),
    });
  }
}
