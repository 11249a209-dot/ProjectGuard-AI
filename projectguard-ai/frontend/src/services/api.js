const API_ROOT = `${import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000'}/api`

async function request(path, options = {}) {
  const token = localStorage.getItem('projectguard-token')
  const response = await fetch(`${API_ROOT}${path}`, {
    ...options,
    headers: {
      ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    throw new Error(payload.error || `Request failed (${response.status})`)
  }
  return response
}

export async function apiJson(path, options) {
  const response = await request(path, options)
  return response.json()
}

export async function uploadProject(formData, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_ROOT}/projects/upload`)
    const token = localStorage.getItem('projectguard-token')
    if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`)
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 100))
    }
    xhr.onload = () => {
      const payload = JSON.parse(xhr.responseText || '{}')
      if (xhr.status >= 200 && xhr.status < 300) resolve(payload)
      else reject(new Error(payload.error || `Upload failed (${xhr.status})`))
    }
    xhr.onerror = () => reject(new Error('Could not reach the ProjectGuard API.'))
    xhr.send(formData)
  })
}

export function downloadReport(id) {
  return request(`/reports/${id}/pdf`).then((response) => response.blob())
}
