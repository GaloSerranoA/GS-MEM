import { type ComponentType, type FormEvent, useEffect, useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { HashRouter, Link, NavLink, Route, Routes, useParams } from 'react-router-dom'
import type { ForceGraphProps } from 'react-force-graph-2d'
import type { Backlink, PageDetail, PageSummary, SearchResult, Triple } from './api'
import {
  findTriples,
  getBacklinks,
  getPage,
  listPages,
  putPage,
  searchPages,
} from './api'

type LoadState<T> =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; data: T }

interface GraphNode {
  id: string
  label: string
  kind: 'subject' | 'object' | 'both'
  x?: number
  y?: number
}

interface GraphLink {
  source: string | GraphNode
  target: string | GraphNode
  predicate: string
  confidence?: number | null
}

type GraphData = {
  nodes: GraphNode[]
  links: GraphLink[]
}

type PositionedGraphNode = GraphNode & { x: number; y: number }

type ForceGraphComponent = ComponentType<ForceGraphProps<GraphNode, GraphLink>>

const DEFAULT_PAGE_LIMIT = 100
const DEFAULT_SEARCH_LIMIT = 20

export default function App() {
  return (
    <HashRouter>
      <div className="app-shell">
        <header className="app-header">
          <Link className="brand" to="/">
            GS-MEM
          </Link>
          <nav aria-label="Primary">
            <NavLink to="/" end>
              Pages
            </NavLink>
            <NavLink to="/graph">Graph</NavLink>
          </nav>
        </header>
        <Routes>
          <Route path="/" element={<PagesListView />} />
          <Route path="/page/:slug" element={<PageDetailView />} />
          <Route path="/graph" element={<GraphView />} />
          <Route path="*" element={<NotFoundView />} />
        </Routes>
        <footer className="app-footer">
          <span>
            Created by <strong>Galo Serrano Abad</strong>
          </span>
          <span className="app-footer-org">NANTAR AI ROBOTICS</span>
        </footer>
      </div>
    </HashRouter>
  )
}

function PagesListView() {
  const [pages, setPages] = useState<LoadState<PageSummary[]>>({ status: 'loading' })
  const [tag, setTag] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<LoadState<SearchResult[]> | null>(null)

  useEffect(() => {
    let cancelled = false
    setPages({ status: 'loading' })
    listPages({ tag: tag.trim() || undefined, limit: DEFAULT_PAGE_LIMIT })
      .then((data) => {
        if (!cancelled) setPages({ status: 'ready', data })
      })
      .catch((error: unknown) => {
        if (!cancelled) setPages({ status: 'error', message: messageFrom(error) })
      })
    return () => {
      cancelled = true
    }
  }, [tag])

  function handleSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const q = searchQuery.trim()
    if (!q) {
      setSearchResults(null)
      return
    }
    setSearchResults({ status: 'loading' })
    searchPages(q, DEFAULT_SEARCH_LIMIT)
      .then((data) => setSearchResults({ status: 'ready', data }))
      .catch((error: unknown) =>
        setSearchResults({ status: 'error', message: messageFrom(error) }),
      )
  }

  return (
    <main className="layout">
      <section className="section-heading">
        <h1>Pages</h1>
        <p>Browse durable memory pages and search compiled page text.</p>
      </section>

      <section className="toolbar" aria-label="Page filters">
        <label>
          Tag
          <input
            value={tag}
            onChange={(event) => setTag(event.target.value)}
            placeholder="optional tag"
          />
        </label>
        <form className="search-form" onSubmit={handleSearch}>
          <label>
            Search
            <input
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder="query"
            />
          </label>
          <button type="submit">Search</button>
        </form>
      </section>

      {searchResults && (
        <section className="panel">
          <h2>Search Results</h2>
          <LoadBoundary state={searchResults}>
            {(results) =>
              results.length === 0 ? (
                <EmptyState message="No matching pages." />
              ) : (
                <div className="result-list">
                  {results.map((result) => (
                    <article className="result-item" key={result.slug}>
                      <Link to={`/page/${encodeURIComponent(result.slug)}`}>{result.slug}</Link>
                      <p>{result.snippet}</p>
                    </article>
                  ))}
                </div>
              )
            }
          </LoadBoundary>
        </section>
      )}

      <section className="panel">
        <h2>All Pages</h2>
        <LoadBoundary state={pages}>
          {(items) =>
            items.length === 0 ? (
              <EmptyState message="No pages found." />
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Slug</th>
                      <th>Title</th>
                      <th>Scope</th>
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((page) => (
                      <tr key={page.slug}>
                        <td>
                          <Link to={`/page/${encodeURIComponent(page.slug)}`}>{page.slug}</Link>
                        </td>
                        <td>{page.title || 'Untitled'}</td>
                        <td>
                          <ScopeBadge scope={page.scope} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          }
        </LoadBoundary>
      </section>
    </main>
  )
}

function PageDetailView() {
  const { slug } = useParams()
  const decodedSlug = slug ? decodeURIComponent(slug) : ''
  const [page, setPage] = useState<LoadState<PageDetail>>({ status: 'loading' })
  const [backlinks, setBacklinks] = useState<LoadState<Backlink[]>>({ status: 'loading' })
  const [isEditing, setIsEditing] = useState(false)
  const [editBody, setEditBody] = useState('')
  const [editTitle, setEditTitle] = useState('')
  const [editScope, setEditScope] = useState('')
  const [saveMessage, setSaveMessage] = useState<string | null>(null)

  useEffect(() => {
    if (!decodedSlug) return
    refreshPage(decodedSlug)
  }, [decodedSlug])

  function refreshPage(nextSlug = decodedSlug) {
    setPage({ status: 'loading' })
    setBacklinks({ status: 'loading' })
    setSaveMessage(null)

    getPage(nextSlug)
      .then((data) => {
        setPage({ status: 'ready', data })
        setEditBody(data.compiled_truth)
        setEditTitle(data.title ?? '')
        setEditScope(data.scope ?? '')
      })
      .catch((error: unknown) => setPage({ status: 'error', message: messageFrom(error) }))

    getBacklinks(nextSlug)
      .then((data) => setBacklinks({ status: 'ready', data }))
      .catch((error: unknown) => setBacklinks({ status: 'error', message: messageFrom(error) }))
  }

  function handleSave(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setSaveMessage('Saving...')
    putPage(decodedSlug, {
      body: editBody,
      title: optionalText(editTitle),
      scope: optionalText(editScope),
    })
      .then(() => {
        setIsEditing(false)
        setSaveMessage('Saved.')
        refreshPage()
      })
      .catch((error: unknown) => setSaveMessage(messageFrom(error)))
  }

  if (!decodedSlug) {
    return (
      <main className="layout">
        <EmptyState message="Missing page slug." />
      </main>
    )
  }

  return (
    <main className="layout">
      <LoadBoundary state={page}>
        {(data) => (
          <>
            <section className="section-heading detail-heading">
              <div>
                <Link className="subtle-link" to="/">
                  Pages
                </Link>
                <h1>{data.title || data.slug}</h1>
                <p>
                  {data.slug} <ScopeBadge scope={data.scope} />
                </p>
              </div>
              <button type="button" onClick={() => setIsEditing((value) => !value)}>
                {isEditing ? 'Cancel' : 'Edit'}
              </button>
            </section>

            {isEditing ? (
              <form className="panel edit-form" onSubmit={handleSave}>
                <label>
                  Title
                  <input value={editTitle} onChange={(event) => setEditTitle(event.target.value)} />
                </label>
                <label>
                  Scope
                  <input value={editScope} onChange={(event) => setEditScope(event.target.value)} />
                </label>
                <label>
                  Body
                  <textarea value={editBody} onChange={(event) => setEditBody(event.target.value)} />
                </label>
                <div className="form-actions">
                  <button type="submit">Save</button>
                  {saveMessage && <span>{saveMessage}</span>}
                </div>
              </form>
            ) : (
              <article className="panel markdown-body">
                <ReactMarkdown>{data.compiled_truth || '_Empty page._'}</ReactMarkdown>
              </article>
            )}

            <section className="two-column">
              <JsonPanel title="Frontmatter" value={data.frontmatter} />
              <JsonPanel title="Timeline" value={data.timeline} />
            </section>

            <section className="panel">
              <h2>Backlinks</h2>
              <LoadBoundary state={backlinks}>
                {(items) =>
                  items.length === 0 ? (
                    <EmptyState message="No backlinks." />
                  ) : (
                    <ul className="link-list">
                      {items.map((item) => (
                        <li key={`${item.from_slug}:${item.edge_type}`}>
                          <Link to={`/page/${encodeURIComponent(item.from_slug)}`}>
                            {item.from_slug}
                          </Link>
                          <span>{item.edge_type}</span>
                        </li>
                      ))}
                    </ul>
                  )
                }
              </LoadBoundary>
            </section>
          </>
        )}
      </LoadBoundary>
    </main>
  )
}

function GraphView() {
  const [mode, setMode] = useState<'subject' | 'object'>('subject')
  const [term, setTerm] = useState('')
  const [triples, setTriples] = useState<LoadState<Triple[]> | null>(null)

  const graphData = useMemo(() => {
    return triples?.status === 'ready' ? graphFromTriples(triples.data) : emptyGraph()
  }, [triples])

  function handleLoad(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const value = term.trim()
    if (!value) return

    setTriples({ status: 'loading' })
    findTriples(mode === 'subject' ? { subject: value } : { object: value })
      .then((data) => setTriples({ status: 'ready', data }))
      .catch((error: unknown) => setTriples({ status: 'error', message: messageFrom(error) }))
  }

  return (
    <main className="layout graph-layout">
      <section className="section-heading">
        <h1>Graph</h1>
        <p>Load triples by subject or object and inspect predicate-labeled edges.</p>
      </section>

      <form className="toolbar" onSubmit={handleLoad}>
        <fieldset className="segmented">
          <legend>Lookup</legend>
          <label>
            <input
              type="radio"
              name="lookup"
              checked={mode === 'subject'}
              onChange={() => setMode('subject')}
            />
            Subject
          </label>
          <label>
            <input
              type="radio"
              name="lookup"
              checked={mode === 'object'}
              onChange={() => setMode('object')}
            />
            Object
          </label>
        </fieldset>
        <label>
          Value
          <input
            value={term}
            onChange={(event) => setTerm(event.target.value)}
            placeholder={mode === 'subject' ? 'alice' : 'bob'}
          />
        </label>
        <button type="submit">Load</button>
      </form>

      {triples?.status === 'error' && <ErrorState message={triples.message} />}
      {triples?.status === 'loading' && <LoadingState />}

      <section className="graph-panel">
        {triples?.status === 'ready' && triples.data.length === 0 ? (
          <EmptyState message="No triples found." />
        ) : (
          <ClientForceGraph data={graphData} />
        )}
      </section>
    </main>
  )
}

function ClientForceGraph({ data }: { data: GraphData }) {
  const [ForceGraph, setForceGraph] = useState<ForceGraphComponent | null>(null)

  useEffect(() => {
    let cancelled = false
    import('react-force-graph-2d').then((module) => {
      if (!cancelled) {
        setForceGraph(() => module.default as unknown as ForceGraphComponent)
      }
    })
    return () => {
      cancelled = true
    }
  }, [])

  if (!ForceGraph) {
    return <LoadingState />
  }

  return (
    <ForceGraph
      graphData={data}
      width={900}
      height={520}
      backgroundColor="#ffffff"
      nodeId="id"
      nodeLabel="label"
      nodeColor={(node) =>
        node.kind === 'both' ? '#8b5cf6' : node.kind === 'subject' ? '#6366f1' : '#14b8a6'
      }
      linkColor={() => '#c7cdda'}
      linkDirectionalArrowLength={5}
      linkDirectionalArrowRelPos={1}
      linkLabel={(link) => link.predicate}
      linkCanvasObjectMode={() => 'after'}
      linkCanvasObject={drawLinkLabel}
    />
  )
}

function LoadBoundary<T>({
  state,
  children,
}: {
  state: LoadState<T>
  children: (data: T) => JSX.Element
}) {
  if (state.status === 'loading') return <LoadingState />
  if (state.status === 'error') return <ErrorState message={state.message} />
  return children(state.data)
}

function LoadingState() {
  return <p className="state-text">Loading...</p>
}

function ErrorState({ message }: { message: string }) {
  return <p className="state-text error">Error: {message}</p>
}

function EmptyState({ message }: { message: string }) {
  return <p className="state-text empty">{message}</p>
}

function ScopeBadge({ scope }: { scope?: string | null }) {
  const value = scope || 'default'
  return <span className={`badge badge-${value}`}>{value}</span>
}

function JsonPanel({ title, value }: { title: string; value: unknown }) {
  return (
    <section className="panel">
      <h2>{title}</h2>
      <pre>{formatJson(value)}</pre>
    </section>
  )
}

function NotFoundView() {
  return (
    <main className="layout">
      <EmptyState message="View not found." />
    </main>
  )
}

function graphFromTriples(triples: Triple[]): GraphData {
  const nodes = new Map<string, GraphNode>()
  const links: GraphLink[] = []

  for (const triple of triples) {
    upsertNode(nodes, triple.subject, 'subject')
    upsertNode(nodes, triple.object, 'object')
    links.push({
      source: triple.subject,
      target: triple.object,
      predicate: triple.predicate,
      confidence: triple.confidence,
    })
  }

  return { nodes: [...nodes.values()], links }
}

function upsertNode(nodes: Map<string, GraphNode>, id: string, kind: GraphNode['kind']) {
  const existing = nodes.get(id)
  if (!existing) {
    nodes.set(id, { id, label: id, kind })
    return
  }
  if (existing.kind !== kind) {
    existing.kind = 'both'
  }
}

function emptyGraph(): GraphData {
  return { nodes: [], links: [] }
}

function drawLinkLabel(link: GraphLink, context: CanvasRenderingContext2D, globalScale: number) {
  const source = positionedNode(link.source)
  const target = positionedNode(link.target)
  if (!source || !target) return

  const x = (source.x + target.x) / 2
  const y = (source.y + target.y) / 2
  const fontSize = 11 / globalScale
  const padding = 3 / globalScale
  const textWidth = context.measureText(link.predicate).width

  context.save()
  context.font = `${fontSize}px system-ui, sans-serif`
  context.fillStyle = 'rgba(255, 255, 255, 0.88)'
  context.fillRect(x - textWidth / 2 - padding, y - fontSize, textWidth + padding * 2, fontSize * 1.35)
  context.fillStyle = '#334155'
  context.textAlign = 'center'
  context.textBaseline = 'middle'
  context.fillText(link.predicate, x, y - fontSize * 0.25)
  context.restore()
}

function positionedNode(value: GraphLink['source']): PositionedGraphNode | null {
  if (typeof value !== 'object' || value === null) return null
  return typeof value.x === 'number' && typeof value.y === 'number'
    ? (value as PositionedGraphNode)
    : null
}

function optionalText(value: string): string | undefined {
  const trimmed = value.trim()
  return trimmed ? trimmed : undefined
}

function formatJson(value: unknown): string {
  if (value === null || value === undefined) return '{}'
  return JSON.stringify(value, null, 2)
}

function messageFrom(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
