import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './projectguard.css'
import './faculty-compare.css'
import './landing-tech.css'
import './report-mode.css'
import './landing-auth-links.css'
import App from './ProjectGuard.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
