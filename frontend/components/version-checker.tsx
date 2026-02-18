
"use client"
import { useEffect, useRef } from "react"

export function VersionChecker() {
  const versionRef = useRef<string | null>(null)
  
  useEffect(() => {
    const checkVersion = async () => {
      try {
        const res = await fetch("/api/version", { cache: "no-store" })
        const data = await res.json()
        if (versionRef.current === null) {
          versionRef.current = data.version
        } else if (versionRef.current !== data.version) {
          // New version available - reload
          window.location.reload()
        }
      } catch {}
    }
    
    // Check on mount
    checkVersion()
    
    // Check when tab becomes visible
    const onVisibility = () => {
      if (document.visibilityState === "visible") checkVersion()
    }
    document.addEventListener("visibilitychange", onVisibility)
    
    // Check every 60 seconds
    const interval = setInterval(checkVersion, 60000)
    
    return () => {
      document.removeEventListener("visibilitychange", onVisibility)
      clearInterval(interval)
    }
  }, [])
  
  return null
}
