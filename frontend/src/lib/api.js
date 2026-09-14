// Thin fetch wrapper around the MyEchoMind FastAPI backend.
// All calls go through /api, which vite.config.js proxies to :8000 in dev.
// In production, host the built assets behind the same origin as the API.

const BASE = '/api'

async function request(path, { method = 'GET', body, headers, isForm = false } = {}) {
  const opts = { method, headers: { ...(headers || {}) } }
  if (body !== undefined) {
    if (isForm) {
      opts.body = body
    } else {
      opts.headers['Content-Type'] = 'application/json'
      opts.body = JSON.stringify(body)
    }
  }
  const resp = await fetch(`${BASE}${path}`, opts)
  const text = await resp.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = text
  }
  if (!resp.ok) {
    const detail = (data && data.detail) || resp.statusText || 'request failed'
    throw new Error(`${resp.status} ${detail}`)
  }
  return data
}

export const api = {
  health: () => request('/health'),
  chat: ({ message, userId, conversationId, useRag = true }) =>
    request('/chat', {
      method: 'POST',
      body: {
        message,
        user_id: userId || 'web',
        conversation_id: conversationId || null,
        use_rag: useRag,
      },
    }),
  search: ({ query, topK = 5, rewriteN = 3, rerank = true }) =>
    request('/search', {
      method: 'POST',
      body: { query, top_k: topK, rewrite_n: rewriteN, rerank },
    }),
  knowledgeStats: () => request('/knowledge/stats'),
  knowledgeAdd: (documents) =>
    request('/knowledge/add', { method: 'POST', body: { documents } }),
  knowledgeUpload: (file) => {
    const form = new FormData()
    form.append('file', file)
    return request('/knowledge/upload', { method: 'POST', body: form, isForm: true })
  },
  skills: () => request('/skills'),
  skillsReload: () => request('/skills/reload', { method: 'POST' }),
  monitor: () => request('/monitor'),
  evalRun: (cases) =>
    request('/eval/run', {
      method: 'POST',
      body: cases ? { cases } : {},
    }),
}
