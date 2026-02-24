"use client"
import { useEffect, useRef } from "react"

export function VersionChecker() {
  const versionRef = useRef<string | null>(null)
  
  useEffect(() => {
    const checkVersion = async () => {
      try {
        const res = await fetch("/api/version?t=" + Date.now(), { cache: "no-store" })
        const data = await res.json()
        if (versionRef.current === null) {
          versionRef.current = data.version
        } else if (versionRef.current !== data.version) {
          // Force reload bypassing cache
          window.location.href = window.location.pathname + "?v=" + Date.now()
        }
      } catch {}
    }
    
    checkVersion()
    
    const onVisibility = () => {
      if (document.visibilityState === "visible") checkVersion()
    }
    document.addEventListener("visibilitychange", onVisibility)
    
    const interval = setInterval(checkVersion, 15000)
    
    return () => {
      document.removeEventListener("visibilitychange", onVisibility)
      clearInterval(interval)
    }
  }, [])
  
  return null
}
