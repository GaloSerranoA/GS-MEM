export interface PageSummary {
  slug: string
  scope?: string | null
  title?: string | null
}

export interface PageDetail {
  slug: string
  scope?: string | null
  title?: string | null
  compiled_truth: string
  timeline: unknown
  frontmatter: unknown
}

export interface SearchResult {
  slug: string
  snippet: string
}

export interface Backlink {
  from_slug: string
  edge_type: string
}

export interface Triple {
  id: number
  subject: string
  predicate: string
  object: string
  confidence?: number | null
  is_current: boolean
}

export interface PutPageBody {
  body: string
  title?: string
  scope?: string
  frontmatter?: unknown
}

export interface PutPageResult {
  ok: boolean
  slug: string
}

export interface AddTripleBody {
  subject: string
  predicate: string
  object: string
  confidence?: number
}

export interface AddTripleResult {
  id: number
}

interface RequestOptions extends RequestInit {
  body?: BodyInit | null
}

async function apiRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers = new Headers(options.headers)
  if (options.body && !headers.has('content-type')) {
    headers.set('content-type', 'application/json')
  }

  const response = await fetch(path, { ...options, headers })
  if (!response.ok) {
    const message = await errorMessage(response)
    throw new Error(message)
  }

  return (await response.json()) as T
}

async function errorMessage(response: Response): Promise<string> {
  const fallback = `${response.status} ${response.statusText}`
  try {
    const data = (await response.json()) as { error?: unknown }
    return typeof data.error === 'string' ? data.error : fallback
  } catch {
    return fallback
  }
}

function queryString(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== '') {
      search.set(key, String(value))
    }
  }
  const encoded = search.toString()
  return encoded ? `?${encoded}` : ''
}

export function listPages(params: { tag?: string; limit?: number } = {}): Promise<PageSummary[]> {
  return apiRequest<PageSummary[]>(`/api/pages${queryString(params)}`)
}

export function getPage(slug: string): Promise<PageDetail> {
  return apiRequest<PageDetail>(`/api/pages/${encodeURIComponent(slug)}`)
}

export function putPage(slug: string, body: PutPageBody): Promise<PutPageResult> {
  return apiRequest<PutPageResult>(`/api/pages/${encodeURIComponent(slug)}`, {
    method: 'PUT',
    body: JSON.stringify(body),
  })
}

export function getBacklinks(slug: string): Promise<Backlink[]> {
  return apiRequest<Backlink[]>(`/api/pages/${encodeURIComponent(slug)}/backlinks`)
}

export function searchPages(q: string, limit?: number): Promise<SearchResult[]> {
  return apiRequest<SearchResult[]>(`/api/search${queryString({ q, limit })}`)
}

export function findTriples(params: { subject?: string; object?: string }): Promise<Triple[]> {
  return apiRequest<Triple[]>(`/api/triples${queryString(params)}`)
}

export function addTriple(body: AddTripleBody): Promise<AddTripleResult> {
  return apiRequest<AddTripleResult>('/api/triples', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}
