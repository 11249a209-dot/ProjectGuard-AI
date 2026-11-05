import { useState } from 'react'
import { ArrowLeft } from 'lucide-react'
import './simple-login.css'

export default function SimpleLogin({ onLogin, onBack, busy }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')

  async function submit(event) {
    event.preventDefault()
    setError('')
    try {
      await onLogin({ email, password })
    } catch (loginError) {
      setError(loginError.message || 'Unable to sign in.')
    }
  }

  return <main className="simple-login">
    <button className="login-back" type="button" aria-label="Back" title="Back" onClick={onBack}><ArrowLeft size={19} /></button>
    <form onSubmit={submit}>
      <input aria-label="Email Address" type="email" autoComplete="username" placeholder="Email Address" value={email} onChange={(event) => setEmail(event.target.value)} required />
      <input aria-label="Password" type="password" autoComplete="current-password" placeholder="Password" value={password} onChange={(event) => setPassword(event.target.value)} required />
      {error && <p className="login-error" role="alert">{error}</p>}
      <button type="submit" disabled={busy}>{busy ? 'Signing in…' : 'Login'}</button>
    </form>
  </main>
}
