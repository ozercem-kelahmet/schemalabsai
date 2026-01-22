"use client"

import { useState, useEffect, useCallback } from "react"
import { useSearchParams, useRouter } from "next/navigation"
import { ConfigStep } from "@/components/build/config-step"
import { TrainingStep } from "@/components/build/training-step"
import { EvaluateStep } from "@/components/build/evaluate-step"
import { mockDatasets, generateTrainingProgress, mockModels } from "@/lib/mock-data"
import type { Dataset, SyncMode, TrainingMetrics, EvaluationMetrics, Model } from "@/lib/types"
import { Check } from "lucide-react"

type Step = "config" | "training" | "evaluate"
type TrainingStatus = "initializing" | "training" | "paused" | "completing"

const steps = [
  { id: "config", label: "Configure" },
  { id: "training", label: "Build" },
  { id: "evaluate", label: "Evaluate" },
]

export function BuildWizard() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const initialDatasetId = searchParams.get("dataset")

  const [selectedDatasets, setSelectedDatasets] = useState<Dataset[]>(() => {
    if (initialDatasetId) {
      const ds = mockDatasets.find((d) => d.id === initialDatasetId)
      return ds ? [ds] : []
    }
    return []
  })

  const [currentStep, setCurrentStep] = useState<Step>("config")
  const [modelName, setModelName] = useState("")
  const [modelDescription, setModelDescription] = useState("")
  const [syncMode, setSyncMode] = useState<SyncMode>("manual")
  const [baseModel, setBaseModel] = useState<string>("schema-v0")

  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>("initializing")
  const [currentMetrics, setCurrentMetrics] = useState<TrainingMetrics | null>(null)
  const [metricsHistory, setMetricsHistory] = useState<TrainingMetrics[]>([])
  const [logs, setLogs] = useState<string[]>([])
  const [elapsedTime, setElapsedTime] = useState(0)
  const [isPaused, setIsPaused] = useState(false)

  const [evalMetrics, setEvalMetrics] = useState<EvaluationMetrics | null>(null)
  const [builtModel, setBuiltModel] = useState<Model | null>(null)

  const totalEpochs = 50

  const handleDatasetToggle = (dataset: Dataset) => {
    setSelectedDatasets((prev) => {
      const isSelected = prev.some((d) => d.id === dataset.id)
      if (isSelected) {
        return prev.filter((d) => d.id !== dataset.id)
      } else {
        return [...prev, dataset]
      }
    })
  }

  const addLog = useCallback((message: string) => {
    setLogs((prev) => [...prev, message])
  }, [])

  const startTraining = () => {
    setCurrentStep("training")
    setTrainingStatus("initializing")
    setMetricsHistory([])
    setLogs([])
    setElapsedTime(0)

    addLog("Initializing build environment...")
    addLog(`Model: ${modelName}`)
    addLog(`Base Model: ${baseModel}`)
    addLog(`Sync Mode: ${syncMode}`)
    addLog(`Connecting ${selectedDatasets.length} data source(s)...`)
    selectedDatasets.forEach((ds) => {
      addLog(`  → ${ds.name} (${ds.source}): ${ds.rows.toLocaleString()} rows, ${ds.columns} columns`)
    })

    setTimeout(() => {
      addLog("Data preprocessing complete")
      addLog("Building knowledge base...")
      addLog("Training neural architecture...")
      setTrainingStatus("training")
    }, 2000)
  }

  useEffect(() => {
    if (trainingStatus !== "training" || isPaused) return

    const epochInterval = setInterval(() => {
      setCurrentMetrics((prev) => {
        const currentEpoch = prev ? prev.epoch + 1 : 1

        if (currentEpoch > totalEpochs) {
          clearInterval(epochInterval)
          setTrainingStatus("completing")
          addLog("Build complete!")
          addLog("Evaluating model performance...")

          setTimeout(() => {
            const finalAccuracy = 0.9 + Math.random() * 0.05
            setEvalMetrics({
              accuracy: finalAccuracy,
              precision: finalAccuracy - 0.02 + Math.random() * 0.04,
              recall: finalAccuracy - 0.03 + Math.random() * 0.05,
              f1Score: finalAccuracy - 0.01 + Math.random() * 0.02,
            })

            const newModel: Model = {
              id: `model-${Date.now()}`,
              name: modelName,
              description: modelDescription,
              datasets: selectedDatasets.map((ds) => ({
                datasetId: ds.id,
                datasetName: ds.name,
                source: ds.source,
                rows: ds.rows,
                columns: ds.columns,
                connectedAt: new Date(),
                lastSynced: new Date(),
                syncStatus: ds.syncStatus || "synced",
              })),
              syncMode,
              baseModel,
              accuracy: finalAccuracy,
              createdAt: new Date(),
              updatedAt: new Date(),
              status: "completed",
            }
            setBuiltModel(newModel)
            mockModels.push(newModel)

            setCurrentStep("evaluate")
          }, 1500)

          return prev
        }

        const newMetrics = generateTrainingProgress(currentEpoch, totalEpochs)
        setMetricsHistory((h) => [...h, newMetrics])

        if (currentEpoch % 5 === 0) {
          addLog(
            `Epoch ${currentEpoch}/${totalEpochs} - Loss: ${newMetrics.loss.toFixed(4)}, Accuracy: ${(newMetrics.accuracy * 100).toFixed(1)}%`,
          )
        }

        return newMetrics
      })
    }, 300)

    return () => clearInterval(epochInterval)
  }, [
    trainingStatus,
    isPaused,
    totalEpochs,
    addLog,
    selectedDatasets,
    modelName,
    modelDescription,
    syncMode,
    baseModel,
  ])

  useEffect(() => {
    if (trainingStatus !== "training" || isPaused) return

    const timer = setInterval(() => {
      setElapsedTime((t) => t + 1)
    }, 1000)

    return () => clearInterval(timer)
  }, [trainingStatus, isPaused])

  const handlePause = () => {
    setIsPaused(true)
    setTrainingStatus("paused")
    addLog("Build paused")
  }

  const handleResume = () => {
    setIsPaused(false)
    setTrainingStatus("training")
    addLog("Build resumed")
  }

  const handleStop = () => {
    setCurrentStep("config")
    setTrainingStatus("initializing")
    setCurrentMetrics(null)
    setMetricsHistory([])
    setLogs([])
    setIsPaused(false)
  }

  const handleTrainAgain = () => {
    setCurrentStep("config")
    setEvalMetrics(null)
    setBuiltModel(null)
  }

  const handleOpenPlayground = () => {
    if (builtModel) {
      router.push(`/playground?model=${builtModel.id}`)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-center">
        <div className="flex items-center gap-2">
          {steps.map((step, index) => {
            const isComplete =
              (step.id === "config" && currentStep !== "config") ||
              (step.id === "training" && currentStep === "evaluate")
            const isCurrent = step.id === currentStep

            return (
              <div key={step.id} className="flex items-center">
                <div className="flex items-center gap-2">
                  <div
                    className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-medium transition-colors ${
                      isComplete
                        ? "bg-emerald-500 text-white"
                        : isCurrent
                          ? "bg-[#0052CC] text-white"
                          : "bg-white/10 text-gray-400"
                    }`}
                  >
                    {isComplete ? <Check className="h-4 w-4" /> : index + 1}
                  </div>
                  <span className={`text-sm font-medium ${isCurrent ? "text-white" : "text-gray-400"}`}>
                    {step.label}
                  </span>
                </div>
                {index < steps.length - 1 && <div className="mx-4 h-px w-16 bg-white/10" />}
              </div>
            )
          })}
        </div>
      </div>

      {currentStep === "config" && (
        <ConfigStep
          selectedDatasets={selectedDatasets}
          modelName={modelName}
          modelDescription={modelDescription}
          syncMode={syncMode}
          baseModel={baseModel}
          onDatasetToggle={handleDatasetToggle}
          onModelNameChange={setModelName}
          onModelDescriptionChange={setModelDescription}
          onSyncModeChange={setSyncMode}
          onBaseModelChange={setBaseModel}
          onStartTraining={startTraining}
        />
      )}

      {currentStep === "training" && (
        <TrainingStep
          currentMetrics={currentMetrics}
          history={metricsHistory}
          logs={logs}
          status={trainingStatus}
          elapsedTime={elapsedTime}
          onPause={handlePause}
          onResume={handleResume}
          onStop={handleStop}
        />
      )}

      {currentStep === "evaluate" && evalMetrics && builtModel && (
        <EvaluateStep
          metrics={evalMetrics}
          model={builtModel}
          trainingTime={elapsedTime}
          onTrainAgain={handleTrainAgain}
          onOpenPlayground={handleOpenPlayground}
        />
      )}
    </div>
  )
}
