"use client"

import { useEffect } from "react"

export function VersionCheck() {
  useEffect(() => {
    const currentBuild = document.querySelector('script#__NEXT_DATA__')
    if (!currentBuild) return
    try {
      const data = JSON.parse(currentBuild.textContent || "{}")
      const buildId = data.buildId
      if (!buildId) return
      const stored = localStorage.getItem("next_build_id")
      if (stored && stored !== buildId) {
        localStorage.setItem("next_build_id", buildId)
        window.location.reload()
      }
      localStorage.setItem("next_build_id", buildId)
    } catch {}
  }, [])
  return null
}
