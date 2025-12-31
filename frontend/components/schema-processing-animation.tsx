"use client"

import { useState, useEffect } from "react"
import { cn } from "@/lib/utils"

interface SchemaProcessingAnimationProps {
  isProcessing?: boolean
  onComplete?: () => void
  dataSources?: string[]
  projectName: string
  isTraining?: boolean
  trainingProgress?: {
    epoch: number
    epochs: number
    accuracy: number
    loss: number
    eta?: string
    status?: string
  }
}

export function SchemaProcessingAnimation({
  isProcessing = true,
  onComplete,
  dataSources = [],
  projectName,
  isTraining = false,
  trainingProgress,
}: SchemaProcessingAnimationProps) {
  const [phase, setPhase] = useState(0)
  const [progress, setProgress] = useState(0)
  const [animProgress, setAnimProgress] = useState(0) // Sürekli animasyon için
  const [currentDataSource, setCurrentDataSource] = useState(0)
  const [particles, setParticles] = useState<Array<{ id: number; x: number; y: number; delay: number }>>([])

  const phases = isTraining ? [
    "Preparing training data",
    "Initializing model weights",
    "Training neural network",
    "Optimizing parameters",
    "Validating accuracy",
    "Finalizing model",
  ] : [
    "Initializing Schema DLM",
    "Connecting to data sources",
    "Analyzing table structures",
    "Building data embeddings",
    "Optimizing query pathways",
    "Finalizing neural connections",
  ]

  const dataSourcesLength = dataSources?.length ?? 0

  // Training modunda progress ve phase epoch'a göre
  useEffect(() => {
    if (isTraining && trainingProgress && trainingProgress.epochs > 0) {
      const epochPercent = (trainingProgress.epoch / trainingProgress.epochs) * 100
      setProgress(epochPercent)
      
      const phaseIndex = Math.min(
        Math.floor((trainingProgress.epoch / trainingProgress.epochs) * 6),
        5
      )
      setPhase(phaseIndex)
    }
  }, [isTraining, trainingProgress?.epoch, trainingProgress?.epochs])

  // Sürekli animasyon (yatay/dikey çizgiler ve kareler için)
  useEffect(() => {
    if (!isProcessing) return
    
    const animInterval = setInterval(() => {
      setAnimProgress((prev) => {
        if (prev >= 100) return 0
        return prev + 0.5
      })
    }, 30)
    
    return () => clearInterval(animInterval)
  }, [isProcessing])

  // Normal mod için animasyonlar
  useEffect(() => {
    if (!isProcessing) {
      setPhase(0)
      setProgress(0)
      setCurrentDataSource(0)
      return
    }

    const newParticles = Array.from({ length: 50 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      delay: Math.random() * 2,
    }))
    setParticles(newParticles)

    // Training modunda interval kullanma - epoch'a göre güncelleniyor
    if (isTraining) return

    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(progressInterval)
          return 100
        }
        return prev + 0.5
      })
    }, 30)

    const phaseInterval = setInterval(() => {
      setPhase((prev) => {
        if (prev >= phases.length - 1) {
          clearInterval(phaseInterval)
          return prev
        }
        return prev + 1
      })
    }, 600)

    let dataSourceInterval: NodeJS.Timeout | null = null
    if (dataSourcesLength > 0) {
      dataSourceInterval = setInterval(() => {
        setCurrentDataSource((prev) => {
          if (prev >= dataSourcesLength - 1) {
            return 0
          }
          return prev + 1
        })
      }, 400)
    }

    const completeTimeout = setTimeout(() => {
      onComplete?.()
    }, 4000)

    return () => {
      clearInterval(progressInterval)
      clearInterval(phaseInterval)
      if (dataSourceInterval) clearInterval(dataSourceInterval)
      clearTimeout(completeTimeout)
    }
  }, [isProcessing, dataSourcesLength, onComplete, phases.length, isTraining])

  if (!isProcessing) return null

  return (
    <div className="flex-1 flex items-center justify-center bg-background/95 backdrop-blur-sm">
      <div className="absolute inset-0 overflow-hidden">
        <svg className="absolute inset-0 w-full h-full opacity-20">
          <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path
                d="M 40 0 L 0 0 0 40"
                fill="none"
                stroke="currentColor"
                strokeWidth="0.5"
                className="text-primary"
              />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>

        <div className="absolute inset-0">
          <div
            className="absolute h-[2px] w-full bg-gradient-to-r from-transparent via-primary to-transparent opacity-60"
            style={{
              top: `${((animProgress * 1.2) % 120) - 10}%`,
              transition: "top 0.1s linear",
            }}
          />
          <div
            className="absolute h-full w-[2px] bg-gradient-to-b from-transparent via-primary to-transparent opacity-60"
            style={{
              left: `${((animProgress * 1.5) % 120) - 10}%`,
              transition: "left 0.1s linear",
            }}
          />
        </div>

        {particles.map((particle) => (
          <div
            key={particle.id}
            className="absolute w-1 h-1 rounded-full bg-primary animate-pulse"
            style={{
              left: `${particle.x}%`,
              top: `${particle.y}%`,
              animationDelay: `${particle.delay}s`,
              opacity: 0.4 + Math.random() * 0.4,
              transform: `scale(${0.5 + Math.random() * 1.5})`,
            }}
          />
        ))}

        <div className="absolute inset-0 bg-gradient-radial from-transparent via-transparent to-background" />
      </div>

      <div className="relative z-10 flex flex-col items-center gap-8 px-8 max-w-2xl">
        <div className="relative">
          <div className="absolute -inset-8 animate-spin" style={{ animationDuration: "8s" }}>
            <svg viewBox="0 0 200 200" className="w-full h-full">
              <circle
                cx="100"
                cy="100"
                r="90"
                fill="none"
                stroke="currentColor"
                strokeWidth="1"
                strokeDasharray="8 12"
                className="text-primary/30"
              />
            </svg>
          </div>

          <div className="absolute -inset-4 animate-pulse" style={{ animationDuration: "2s" }}>
            <svg viewBox="0 0 200 200" className="w-full h-full">
              <circle
                cx="100"
                cy="100"
                r="90"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className="text-primary/20"
              />
            </svg>
          </div>

          <div
            className="absolute -inset-2 animate-spin"
            style={{ animationDuration: "4s", animationDirection: "reverse" }}
          >
            <svg viewBox="0 0 200 200" className="w-full h-full">
              <circle
                cx="100"
                cy="100"
                r="90"
                fill="none"
                stroke="currentColor"
                strokeWidth="1"
                strokeDasharray="20 10"
                className="text-primary/40"
              />
            </svg>
          </div>

          <div className="relative w-32 h-32 flex items-center justify-center">
            <div className="absolute inset-0 bg-primary/10 rounded-xl animate-pulse" />
            <svg
              width="64"
              height="64"
              viewBox="0 0 32 32"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              className="relative z-10"
            >
              <rect x="4" y="4" width="8" height="3" fill="currentColor" className="text-primary animate-pulse" />
              <rect
                x="4"
                y="4"
                width="3"
                height="24"
                fill="currentColor"
                className="text-primary animate-pulse"
                style={{ animationDelay: "0.1s" }}
              />
              <rect
                x="4"
                y="25"
                width="8"
                height="3"
                fill="currentColor"
                className="text-primary animate-pulse"
                style={{ animationDelay: "0.2s" }}
              />
              <rect
                x="20"
                y="4"
                width="8"
                height="3"
                fill="currentColor"
                className="text-primary animate-pulse"
                style={{ animationDelay: "0.3s" }}
              />
              <rect
                x="25"
                y="4"
                width="3"
                height="24"
                fill="currentColor"
                className="text-primary animate-pulse"
                style={{ animationDelay: "0.4s" }}
              />
              <rect
                x="20"
                y="25"
                width="8"
                height="3"
                fill="currentColor"
                className="text-primary animate-pulse"
                style={{ animationDelay: "0.5s" }}
              />
              <rect
                x="14"
                y="14"
                width="4"
                height="4"
                fill="currentColor"
                className="text-primary"
                style={{
                  animation: "pulse 0.5s ease-in-out infinite",
                }}
              />
            </svg>

            {[0, 45, 90, 135, 180, 225, 270, 315].map((angle, i) => (
              <div
                key={angle}
                className="absolute w-16 h-[2px] origin-left"
                style={{
                  transform: `rotate(${angle}deg)`,
                  left: "50%",
                  top: "50%",
                }}
              >
                <div
                  className="h-full bg-gradient-to-r from-primary to-transparent animate-pulse"
                  style={{
                    animationDelay: `${i * 0.1}s`,
                    animationDuration: "1s",
                  }}
                />
              </div>
            ))}
          </div>
        </div>

        <div className="text-center">
          <h2 className="text-2xl font-bold text-foreground mb-2 font-sans">{projectName}</h2>
          <p className="text-sm text-muted-foreground font-sans">{isTraining ? phases[phase] : "Preparing your data workspace"}</p>
        </div>

        <div className="flex flex-col items-center gap-4 w-full max-w-md">
          {!isTraining && (
            <div className="flex items-center gap-2 h-6">
              <div className="w-2 h-2 bg-primary rounded-full animate-ping" />
              <span className="text-sm font-medium text-foreground font-sans transition-all duration-300">
                {phases[phase]}
              </span>
            </div>
          )}

          <div className="w-full h-1 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary via-primary to-primary/80 rounded-full transition-all duration-500 ease-out"
              style={{
                width: `${progress}%`,
                boxShadow: "0 0 10px hsl(var(--primary)), 0 0 20px hsl(var(--primary) / 0.5)",
              }}
            />
          </div>

          <span className="text-xs text-muted-foreground font-mono">{Math.round(progress)}%</span>
        </div>

        {isTraining && trainingProgress ? (
          <div className="grid grid-cols-4 gap-3 w-full max-w-lg">
            <div className="p-3 bg-secondary/50 rounded-xl text-center">
              <div className="text-xl font-bold text-foreground">{trainingProgress.epoch}/{trainingProgress.epochs}</div>
              <div className="text-xs text-muted-foreground mt-1">Epoch</div>
            </div>
            <div className="p-3 bg-secondary/50 rounded-xl text-center">
              <div className="text-xl font-bold text-foreground">{trainingProgress.loss.toFixed(3)}</div>
              <div className="text-xs text-muted-foreground mt-1">Loss</div>
            </div>
            <div className="p-3 bg-secondary/50 rounded-xl text-center">
              <div className="text-xl font-bold text-green-500">{trainingProgress.accuracy.toFixed(1)}%</div>
              <div className="text-xs text-muted-foreground mt-1">Accuracy</div>
            </div>
            <div className="p-3 bg-secondary/50 rounded-xl text-center">
              <div className="text-xl font-bold text-blue-500">{trainingProgress.eta || "..."}</div>
              <div className="text-xs text-muted-foreground mt-1">ETA</div>
            </div>
          </div>
        ) : dataSourcesLength > 0 ? (
          <div className="flex flex-wrap justify-center gap-2 max-w-lg">
            {dataSources.map((source, index) => (
              <div
                key={source}
                className={cn(
                  "px-3 py-1.5 rounded-full text-xs font-medium transition-all duration-300 font-sans border",
                  index === currentDataSource
                    ? "bg-primary text-primary-foreground border-primary scale-105 shadow-lg shadow-primary/20"
                    : index < currentDataSource
                      ? "bg-primary/20 text-primary border-primary/30"
                      : "bg-muted text-muted-foreground border-border",
                )}
              >
                {source}
              </div>
            ))}
          </div>
        ) : null}

        <div className="relative w-full max-w-sm">
          <div className="grid grid-cols-4 gap-1">
            {Array.from({ length: 16 }).map((_, i) => (
              <div
                key={i}
                className={cn(
                  "h-6 rounded-sm transition-all duration-200",
                  animProgress > i * 6.25 ? "bg-primary/30" : "bg-muted",
                )}
                style={{
                  transform: animProgress > i * 6.25 ? "scale(1)" : "scale(0.95)",
                }}
              />
            ))}
          </div>
          <div
            className="absolute left-0 right-0 h-[2px] bg-primary shadow-lg shadow-primary/50"
            style={{
              top: `${(animProgress % 100) * 0.24}px`,
              opacity: 0.8,
            }}
          />
        </div>
      </div>

      <svg className="absolute top-8 left-8 w-16 h-16 text-primary/30" viewBox="0 0 64 64">
        <path d="M0 20 L0 0 L20 0" fill="none" stroke="currentColor" strokeWidth="2" />
      </svg>
      <svg className="absolute top-8 right-8 w-16 h-16 text-primary/30" viewBox="0 0 64 64">
        <path d="M44 0 L64 0 L64 20" fill="none" stroke="currentColor" strokeWidth="2" />
      </svg>
      <svg className="absolute bottom-8 left-8 w-16 h-16 text-primary/30" viewBox="0 0 64 64">
        <path d="M0 44 L0 64 L20 64" fill="none" stroke="currentColor" strokeWidth="2" />
      </svg>
      <svg className="absolute bottom-8 right-8 w-16 h-16 text-primary/30" viewBox="0 0 64 64">
        <path d="M44 64 L64 64 L64 44" fill="none" stroke="currentColor" strokeWidth="2" />
      </svg>
    </div>
  )
}
