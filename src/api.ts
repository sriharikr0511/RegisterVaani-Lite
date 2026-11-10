export const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:6007/api'

export async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API}${path}`, options)
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      message = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body)
    } catch { /* retain HTTP status message */ }
    throw new Error(message)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export function jsonOptions(method: string, body?: unknown): RequestInit {
  return { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
}
