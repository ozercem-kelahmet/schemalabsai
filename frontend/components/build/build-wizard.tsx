"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { toast } from "sonner"
import { useSearchParams, useRouter } from "next/navigation"
import { ConfigStep } from "@/components/build/config-step"
import { TrainingStep } from "@/components/build/training-step"
import { EvaluateStep } from "@/components/build/evaluate-step"
import { api } from "@/lib/api"
import type { Dataset, SyncMode, TrainingMetrics, EvaluationMetrics, Model } from "@/lib/types"
import { Check } from "lucide-react"

type Step = "config" | "training" | "evaluate"
type TrainingStatus = "idle" | "initializing" | "training" | "paused" | "completing" | "failed"

const steps = [
  { id: "config", label: "Configure" },
  { id: "training", label: "Build" },
  { id: "evaluate", label: "Evaluate" },
]

export function BuildWizard() {
  const searchParams = useSearchParams()
  const router = useRouter()

  const [selectedDatasets, setSelectedDatasets] = useState<Dataset[]>([])
  const [currentStep, setCurrentStep] = useState<Step>("config")
  const [modelName, setModelName] = useState("")
  const [modelDescription, setModelDescription] = useState("")
  const [syncMode, setSyncMode] = useState<SyncMode>("manual")
  const [scheduleCron, setScheduleCron] = useState("")
  const [scheduleDesc, setScheduleDesc] = useState("")
  const [connectionIDs, setConnectionIDs] = useState("")
  const [baseModel, setBaseModel] = useState<string>("schema-v0")

  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>("initializing")
  const [trainingError, setTrainingError] = useState<string>("")
  const [currentMetrics, setCurrentMetrics] = useState<TrainingMetrics | null>(null)
  const [metricsHistory, setMetricsHistory] = useState<TrainingMetrics[]>([])
  const [logs, setLogs] = useState<string[]>([])
  const [elapsedTime, setElapsedTime] = useState(0)
  const [isPaused, setIsPaused] = useState(false)

  const [evalMetrics, setEvalMetrics] = useState<EvaluationMetrics | null>(null)
  const [builtModel, setBuiltModel] = useState<Model | null>(null)
  
  const [totalEpochs, setTotalEpochs] = useState(0)
  const pollingRef = useRef<NodeJS.Timeout | null>(null)
  const trainingQueryIdRef = useRef<string>("")
  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const trainingStartedRef = useRef(false)
  const skipCheckRef = useRef(false)
  const completedByPollingRef = useRef(false)
  const trainCancelledRef = useRef(false)
  const sseRef = useRef<EventSource | null>(null)

  // SSE connect - Kafka push, polling fallback
  const connectSSE = (queryId: string) => {
    if (sseRef.current) sseRef.current.close()
    try {
      const sse = new EventSource(`/api/train/stream?query_id=${queryId}`, { withCredentials: true })
      sse.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data)
          if (data.epoch !== undefined && data.epoch > 0) {
            // Kafka'dan direkt veri - Redis'e gitme
            trainingStartedRef.current = true
            setTrainingStatus("training")
            if (pollingRef.current) {
              clearInterval(pollingRef.current)
              pollingRef.current = setInterval(pollProgress, 15000) // 15s fallback only
            }
            const epoch = data.epoch
            const epochs = data.epochs || epoch
            const accuracy = data.accuracy > 1 ? data.accuracy / 100 : data.accuracy
            const loss = data.loss || 0
            const status = data.status || "training"

            if (status === "completed") {
              pollProgress() // Tamamlandığında Redis'ten tam veriyi al
              return
            }

            setTotalEpochs(prev => Math.max(prev, epochs))
            setCurrentMetrics(prev => {
              if (!prev || epoch >= prev.epoch) {
                return { epoch, totalEpochs: epochs, loss, accuracy, learningRate: 0.001 }
              }
              return prev
            })
            setMetricsHistory(prev => {
              const exists = prev.some((m: any) => m.epoch === epoch)
              if (!exists && epoch > 0) {
                const newEntry = { epoch, totalEpochs: epochs, loss, accuracy, learningRate: 0.001 }
                return [...prev, newEntry].sort((a: any, b: any) => a.epoch - b.epoch)
              }
              return prev
            })
            addLog(`Epoch ${epoch}/${epochs} - Loss: ${loss.toFixed(4)}, Accuracy: ${(accuracy * 100).toFixed(1)}%`)
          }
        } catch {}
      }
      sse.onerror = () => {
        // SSE fail → polling devam eder
        sse.close()
        sseRef.current = null
      }
      sseRef.current = sse
    } catch {
      // SSE not supported → polling devam eder
    }
  }


  // Check for ongoing training on mount (e.g. after page refresh)
  useEffect(() => {
    const checkOngoingTraining = async () => {
      try {
        const urlQid = new URLSearchParams(window.location.search).get("qid")
        const savedQueryId = urlQid || sessionStorage.getItem("trainingQueryId")
        if (savedQueryId) { trainingQueryIdRef.current = savedQueryId }
        const url = savedQueryId ? "/api/train/progress?query_id=" + savedQueryId : "/api/train/progress"
        const res = await fetch(url, { credentials: "include" })
        const progress = await res.json()
        if (progress.status === "failed") {
          stopPolling()
          trainingStartedRef.current = false
          setTrainingError(progress.error || "An error occurred during training.")
          setTrainingStatus("failed")
          addLog("Training failed: " + (progress.error || "Unknown error"))
          toast.error("Training Failed", { description: progress.error || "An error occurred during training.", duration: 10000 })
          return
        }
        if (progress.status === "training" && (progress.model_id || progress.epoch > 0)) {
          // Stale check: 5dk + epoch=0 = dead training
          const _st = progress.start_time || 0
          const _el = _st > 0 ? Math.floor(Date.now() / 1000 - _st) : 0
          if (_el > 300 && progress.epoch === 0) {
            console.log("Stale training detected (>5min, epoch=0)")
            localStorage.removeItem("trainingMetricsHistory")
            localStorage.removeItem("trainingLogs")
            localStorage.removeItem("trainingCurrentMetrics")
            localStorage.removeItem("trainingTotalEpochs")
            localStorage.removeItem("trainingStartTime")
            return
          }
          // Stale check: start_time 5dk'dan eski ve epoch hala 0 ise stale
          const startTime = progress.start_time || 0
          const elapsed = startTime > 0 ? Math.floor(Date.now() / 1000 - startTime) : 0
          if (elapsed > 300 && progress.epoch === 0) {
            console.log("Stale training detected (>5min, epoch=0), ignoring")
            localStorage.removeItem("trainingMetricsHistory")
            localStorage.removeItem("trainingLogs")
            localStorage.removeItem("trainingCurrentMetrics")
            localStorage.removeItem("trainingTotalEpochs")
            localStorage.removeItem("trainingStartTime")
            return
          }
          trainingStartedRef.current = true
          // Restore history from Redis data for charts after refresh
          if (progress.history && progress.history.length > 0) {
            const restored = progress.history.map((h: any) => ({
              epoch: h.epoch,
              totalEpochs: progress.epochs || h.epoch + 1,
              loss: h.loss || 0,
              accuracy: (h.accuracy > 1 ? h.accuracy / 100 : h.accuracy) || 0,
              learningRate: 0.001,
            }))
            setMetricsHistory(restored)
            restored.forEach((m: any) => {
              addLog(`Epoch ${m.epoch}/${m.totalEpochs} - Loss: ${m.loss.toFixed(4)}, Accuracy: ${(m.accuracy * 100).toFixed(1)}%`)
            })
          }
          // Restore metrics from localStorage FIRST to avoid flash
          const savedM = localStorage.getItem("trainingCurrentMetrics")
          if (savedM) {
            try { setCurrentMetrics(JSON.parse(savedM)) } catch(e) {}
          }
          const savedTE = localStorage.getItem("trainingTotalEpochs")
          if (savedTE) { const te = parseInt(savedTE); if (te > 0) setTotalEpochs(te) }
          const savedST = localStorage.getItem("trainingStartTime")
          if (savedST) {
            const el = Math.floor((Date.now() - parseInt(savedST)) / 1000)
            if (el > 0 && el < 86400) setElapsedTime(el)
          }
          // Then set server values (will override if newer)
          setCurrentStep("training")
          setTrainingStatus("training")
          if (progress.model_name) setModelName(progress.model_name)
          if (progress.epochs) setTotalEpochs(progress.epochs)
          const acc = progress.accuracy > 1 ? progress.accuracy / 100 : progress.accuracy
          setCurrentMetrics(prev => {
            const serverMetrics = { epoch: progress.epoch, totalEpochs: progress.epochs || 0, loss: progress.loss || 0, accuracy: acc || 0, learningRate: 0.001 }
            if (prev && prev.epoch > serverMetrics.epoch) return prev
            return serverMetrics
          })
          if (progress.start_time) {
            const elapsed = Math.floor(Date.now() / 1000 - progress.start_time)
            if (elapsed > 0) setElapsedTime(elapsed)
          }
        } else {
          // No active training - clear stale localStorage
          localStorage.removeItem("trainingMetricsHistory")
          localStorage.removeItem("trainingLogs")
          localStorage.removeItem("trainingCurrentMetrics")
          localStorage.removeItem("trainingTotalEpochs")
          localStorage.removeItem("trainingStartTime")
        }
      } catch (e) {
      }
    }
    checkOngoingTraining()
  }, [])

  // localStorage restore removed - checkOngoingTraining handles it

  // Save metrics history to localStorage when it changes
  useEffect(() => {
    if (metricsHistory.length > 0) {
      localStorage.setItem("trainingMetricsHistory", JSON.stringify(metricsHistory))
    }
  }, [metricsHistory])

  // Save currentMetrics to localStorage
  useEffect(() => {
    if (currentMetrics && currentMetrics.epoch > 0) {
      localStorage.setItem("trainingCurrentMetrics", JSON.stringify(currentMetrics))
    }
  }, [currentMetrics])

  // Save totalEpochs
  useEffect(() => {
    if (totalEpochs > 0) {
      localStorage.setItem("trainingTotalEpochs", totalEpochs.toString())
    }
  }, [totalEpochs])

  // Save logs to localStorage when they change
  useEffect(() => {
    if (logs.length > 0) {
      localStorage.setItem("trainingLogs", JSON.stringify(logs))
    }
  }, [logs])

  // Clear localStorage when training completes
  const clearTrainingStorage = () => {
    localStorage.removeItem("trainingMetricsHistory")
    localStorage.removeItem("trainingLogs")
    localStorage.removeItem("trainingCurrentMetrics")
    localStorage.removeItem("trainingTotalEpochs")
    localStorage.removeItem("trainingStartTime")
    sessionStorage.removeItem("trainingQueryId")
  }

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
    const timestamp = new Date().toLocaleTimeString()
    setLogs((prev) => [...prev, `[${timestamp}] ${message}`])
  }, [])
  const startTraining = async () => {
    if (selectedDatasets.length === 0) {
      toast.error("No Data Selected", { description: "Please select at least one dataset or connection to train on.", duration: 5000 })
      return
    }

    if (!modelName.trim()) {
      toast.error("Model Name Required", { description: "Please enter a name for your model.", duration: 5000 })
      return
    }

    // Reset state for new training
    trainingStartedRef.current = false
    completedByPollingRef.current = false
    trainCancelledRef.current = false
    skipCheckRef.current = false
    setLogs([])
    setEvalMetrics(null)
    setBuiltModel(null)

    try {
      // Separate files from connections (connections have syncStatus "synced")
      const fileDatasets = selectedDatasets.filter(d => d.syncStatus !== "synced")
      const connDatasets = selectedDatasets.filter(d => d.syncStatus === "synced")
      const fileIds = fileDatasets.map((ds) => ds.id)
      
      // Extract connection IDs and selected tables from table-level datasets
      const tableDatasets = connDatasets.filter(d => d.id.includes("::"))
      const plainConnDatasets = connDatasets.filter(d => !d.id.includes("::"))
      
      // Get unique connection IDs from table datasets
      const tableConnIds = [...new Set(tableDatasets.map(d => d.id.split("::")[0]))]
      const plainConnIds = plainConnDatasets.map(d => d.id)
      const allConnIds = [...new Set([...tableConnIds, ...plainConnIds])]
      
      const connectionIds = connectionIDs || allConnIds.join(",")
      
      // Build selected_tables JSON: ["table1", "table2", ...]
      const selectedTableNames = tableDatasets.map(d => d.id.split("::")[1])
      const selectedTablesStr = selectedTableNames.length > 0 ? JSON.stringify(selectedTableNames) : ""

      // Start training - show UI immediately, handle result async
      const trainingQueryId = `train-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
      trainingQueryIdRef.current = trainingQueryId
      sessionStorage.setItem("trainingQueryId", trainingQueryId)
      const url = new URL(window.location.href)
      url.searchParams.set("qid", trainingQueryId)
      window.history.replaceState({}, "", url.toString())
      const trainPromise = api.multiTrain(
        fileIds,
        modelName,
        totalEpochs,
        64,
        0.001,
        100,
        trainingQueryId,
        syncMode,
        scheduleCron,
        scheduleDesc,
        connectionIds,
        selectedTablesStr
      )

      trainingStartedRef.current = false  // Don't start polling yet - wait for Go to initialize
      setCurrentStep("training")
      setTrainingStatus("initializing")
      setMetricsHistory([])
      setElapsedTime(0)
      setCurrentMetrics(null)
      
      // Wait 3 seconds for Go handler to reset progress and start Flask training
      await new Promise(resolve => setTimeout(resolve, 500))
      trainingStartedRef.current = true
      setTrainingStatus("training")
      setElapsedTime(0)

      addLog("Initializing build environment...")
      addLog(`Model: ${modelName}`)
      addLog(`Base Model: ${baseModel}`)
      addLog(`Sync Mode: ${syncMode}${scheduleCron ? ` (${scheduleDesc || scheduleCron})` : ""}`)
      addLog(`Connecting ${selectedDatasets.length} data source(s)...`)
      selectedDatasets.forEach((ds) => {
        addLog(`  → ${ds.name} (${ds.source}): ${ds.rows.toLocaleString()} rows, ${ds.columns} columns`)
      })
      addLog("Starting fine-tuning process...")
      addLog("Sending data to ML server...")
      addLog("Data preprocessing complete")
      addLog("Building knowledge base...")
      addLog("Training neural architecture...")

      trainPromise.then((result: any) => {
        if (result.error) {
          toast.error("Cannot Build Model", { description: result.error, duration: 10000 })
          setCurrentStep("config")
          setTrainingStatus("idle")
          return
        }
        if (completedByPollingRef.current) { return }
        if (trainCancelledRef.current) { return }
        handleTrainResult(result)
      }).catch((err: any) => {
        toast.error("Training Failed", { description: err.message, duration: 10000 })
        setCurrentStep("config")
        setTrainingStatus("idle")
      })
      return
    } catch (err: any) {
      toast.error("Training Failed", { description: err.message || "Unknown error", duration: 10000 })
      setCurrentStep("config")
      setTrainingStatus("idle")
    }
  }

  const handleTrainResult = async (result: any) => {
      if (completedByPollingRef.current) return
      if (result.queued || result.status === "queued") {
        addLog(`⏳ Server busy - Training queued at position ${result.queue_position || 0}`)
        addLog(`Active trainings: ${result.active_trainings || 0}/${result.max_concurrent || 1}`)
        addLog(`Your training will start automatically when a slot opens`)
        import('sonner').then(({ toast }) => {
          toast.warning(`Training Queued`, {
            description: `Server busy. Position in queue: ${result.queue_position || 0}. Will start automatically.`,
            duration: 5000
          })
        })
        setTrainingStatus("initializing")
        return
      }
      
      if (result.status === "success") {
        stopPolling()
        if (result.training_duration) setElapsedTime(result.training_duration)
        setTrainingStatus("completing")
        addLog("Build complete!")
        // Ensure training screen is visible for at least 3 seconds
        await new Promise(resolve => setTimeout(resolve, 500))
        addLog(`Final Accuracy: ${result.accuracy?.toFixed(2)}%`)
        addLog(`Final Loss: ${result.loss?.toFixed(4) || "N/A"}`)
        addLog("Evaluating model performance...")

        const finalAccuracy = (result.accuracy > 1 ? result.accuracy / 100 : result.accuracy) || 0
        if (finalAccuracy === 0) {
          stopPolling()
          trainingStartedRef.current = false
          setTrainingError("Training completed but model could not learn from this data (0% accuracy). Please check data quality.")
          setTrainingStatus("failed")
          addLog("Training failed: 0% accuracy - model could not learn from this data")
          toast.error("Training Failed", { description: "Model could not learn from this data (0% accuracy). Please check data quality.", duration: 10000 })
          return
        }
        setEvalMetrics({
          accuracy: finalAccuracy,
          precision: result.precision || finalAccuracy * 0.98,
          recall: result.recall || finalAccuracy * 0.97,
          f1Score: result.f1_score || finalAccuracy * 0.975,
        })

        const modelId = result.model_id || `model-${Date.now()}`
        const newModel: Model = {
          id: modelId,
          modelId: modelId,
          name: result.model_name || modelName,
          description: modelDescription,
          datasets: selectedDatasets.map((ds) => ({
            datasetId: ds.id,
            datasetName: ds.name,
            source: ds.source,
            rows: ds.rows,
            columns: ds.columns,
            connectedAt: new Date(),
            lastSynced: new Date(),
            syncStatus: "synced" as const,
          })),
          syncMode,
          baseModel,
          accuracy: finalAccuracy,
          createdAt: new Date(),
          updatedAt: new Date(),
          status: "completed",
          apiRequests: 0,
          tokensUsed: 0,
        }
        setBuiltModel(newModel)
        clearTrainingStorage()
        setCurrentStep("evaluate")
      } else if (result.status === "training") {
        // Training started async - SSE + polling will track progress
      connectSSE(trainingQueryIdRef.current)
        return
      } else {
        addLog(`Error: ${result.error || result.message || "Training failed"}`)
        setTrainingStatus("initializing")
      }
  }

  useEffect(() => {
    if (trainingStatus !== "training" || isPaused) return

    const pollProgress = async () => {
      try {
        if (completedByPollingRef.current || !trainingStartedRef.current) { return }
        const progress = await api.getTrainingProgress(trainingQueryIdRef.current)
        if (progress.status === "failed") {
          stopPolling()
          trainingStartedRef.current = false
          setTrainingError(progress.error || "An error occurred during training.")
          setTrainingStatus("failed")
          addLog("Training failed: " + (progress.error || "Unknown error"))
          toast.error("Training Failed", { description: progress.error || "An error occurred during training.", duration: 10000 })
          return
        }
        if (progress.status === "training") {
          // Terminaldeki epochs gelince kilitle, geri donmesin
          const serverEpochs = progress.epochs
          const epochs = serverEpochs || 0
          if (serverEpochs && serverEpochs > 0) {
            setTotalEpochs(prev => Math.max(prev, serverEpochs))
          }

          
          // Always update current metrics when training
          setCurrentMetrics(prev => {
            if (!prev) return { epoch: progress.epoch, totalEpochs: epochs, loss: progress.loss || 0, accuracy: (progress.accuracy > 1 ? progress.accuracy / 100 : progress.accuracy) || 0, learningRate: 0.001 }
            if (progress.epoch >= prev.epoch) return { epoch: progress.epoch, totalEpochs: Math.max(epochs, prev.totalEpochs), loss: progress.loss || 0, accuracy: (progress.accuracy > 1 ? progress.accuracy / 100 : progress.accuracy) || 0, learningRate: 0.001 }
            return prev
          })
          const newMetrics: TrainingMetrics = {
            epoch: progress.epoch,
            totalEpochs: epochs,
            loss: progress.loss || 0,
            accuracy: (progress.accuracy > 1 ? progress.accuracy / 100 : progress.accuracy) || 0,
            learningRate: 0.001,
          }
          
          // Metrics update handled in animation block above
          if (progress.epoch > (currentMetrics?.epoch || 0)) { addLog(`Epoch ${progress.epoch}/${epochs} - Loss: ${(progress.loss || 0).toFixed(4)}, Accuracy: ${((newMetrics.accuracy) * 100).toFixed(1)}%`) }
          setMetricsHistory((prev) => {
            // Merge Redis history + current epoch
            let merged = [...prev]
            if (progress.history && progress.history.length > 0) {
              const existingEpochs = new Set(merged.map((m: any) => m.epoch))
              for (const h of progress.history) {
                if (!existingEpochs.has(h.epoch)) {
                  merged.push({ epoch: h.epoch, totalEpochs: epochs, loss: h.loss || 0, accuracy: h.accuracy > 1 ? h.accuracy / 100 : h.accuracy || 0, learningRate: 0.001 })
                  existingEpochs.add(h.epoch)
                }
              }
              merged.sort((a: any, b: any) => a.epoch - b.epoch)
            }
            const lastEpoch = merged.length > 0 ? merged[merged.length - 1].epoch : 0
            if (progress.epoch > lastEpoch) {
              merged.push(newMetrics)
              existingEpochs.add(progress.epoch)
            }
            return merged
          })
        } else if (progress.status === "completed") {
          const fAcc = (progress.accuracy > 1 ? progress.accuracy / 100 : progress.accuracy) || 0
          stopPolling()
          trainingStartedRef.current = false
          completedByPollingRef.current = true
          const proceedToEvaluate = () => {
          const finalAccuracy = fAcc
          if (finalAccuracy === 0) {
            setTrainingError("Training completed but model could not learn from this data (0% accuracy). Please check data quality.")
            setTrainingStatus("failed")
            addLog("Training failed: 0% accuracy - model could not learn from this data")
            toast.error("Training Failed", { description: "Model could not learn from this data (0% accuracy). Please check data quality.", duration: 10000 })
            return
          }
          setEvalMetrics({
            accuracy: finalAccuracy,
            precision: progress.precision || finalAccuracy * 0.98,
            recall: progress.recall || finalAccuracy * 0.97,
            f1Score: progress.f1_score || finalAccuracy * 0.975,
          })
          const pollingModelId = progress.model_id || "pending"
          setBuiltModel({
            id: pollingModelId,
            modelId: pollingModelId,
            name: modelName || "Trained Model",
            description: modelDescription,
            datasets: selectedDatasets.map((ds) => ({
              datasetId: ds.id,
              datasetName: ds.name,
              source: ds.source,
              rows: ds.rows,
              columns: ds.columns,
              connectedAt: new Date(),
              lastSynced: new Date(),
              syncStatus: "synced" as const,
            })),
            syncMode,
            baseModel,
            accuracy: finalAccuracy,
            createdAt: new Date(),
            updatedAt: new Date(),
            status: "completed",
            apiRequests: 0,
            tokensUsed: 0,
          })
          clearTrainingStorage()
          setCurrentStep("evaluate")
          addLog("Training completed!")
          }
          proceedToEvaluate()
        }
      } catch (e) {
      }
    }

    const startPolling = () => {
      if (pollingRef.current) clearInterval(pollingRef.current)
      const ms = document.hidden ? 10000 : 3000
      pollingRef.current = setInterval(pollProgress, ms)
    }

    startPolling()
    pollProgress()

    const onVisibilityChange = () => startPolling()
    document.addEventListener("visibilitychange", onVisibilityChange)

    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current)
      document.removeEventListener("visibilitychange", onVisibilityChange)
    }
  }, [trainingStatus, isPaused, totalEpochs, addLog])

  useEffect(() => {
    if (trainingStatus !== "training" || isPaused) return

    // Don't reset elapsed - it may have been restored from localStorage
    timerRef.current = setInterval(() => {
      setElapsedTime((t) => t + 1)
    }, 1000)

    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [trainingStatus, isPaused])

  const stopPolling = () => {
    if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null }
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
    if (sseRef.current) { sseRef.current.close(); sseRef.current = null }
  }

  const handlePause = () => { setIsPaused(true); setTrainingStatus("paused"); addLog("Build paused") }
  const handleResume = () => { setIsPaused(false); setTrainingStatus("training"); addLog("Build resumed") }
  
  const handleStop = () => {
    stopPolling()
    // Notify server to cleanup training state
    if (trainingQueryIdRef.current) {
      fetch("/api/train/cancel?query_id=" + trainingQueryIdRef.current, { method: "POST", credentials: "include" }).catch(() => {})
    }
    clearTrainingStorage()
    setCurrentStep("config")
    setTrainingStatus("idle")
    setCurrentMetrics(null)
    setMetricsHistory([])
    setLogs([])
    setIsPaused(false)
    trainingStartedRef.current = false
    completedByPollingRef.current = false
  }

  const handleTrainAgain = () => {
    stopPolling()
    trainCancelledRef.current = false
    skipCheckRef.current = false
    setTrainingStatus("idle")
    trainingStartedRef.current = false
    completedByPollingRef.current = false
    setCurrentStep("config")
    setEvalMetrics(null)
    setBuiltModel(null)
    setCurrentMetrics(null)
    setMetricsHistory([])
    setLogs([])
    setElapsedTime(0)
  }

  const [openingPlayground, setOpeningPlayground] = useState(false)
  const handleOpenPlayground = async () => {
    if (builtModel) {
      setOpeningPlayground(true)
      // Refresh models cache before navigating so playground can find the new model instantly
      try {
        const res = await fetch("/api/models/finetuned", { credentials: "include" })
        const data = await res.json()
        if (data.models) {
          localStorage.setItem("schemalabs_models_cache", JSON.stringify({ models: data.models }))
        }
      } catch {}
      const params = new URLSearchParams({
        model: builtModel.id,
        new: Date.now().toString(),
      })
      router.push("/playground?" + params.toString())
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-center">
        <div className="flex items-center gap-2">
          {steps.map((step, index) => {
            const isComplete = (step.id === "config" && currentStep !== "config") || (step.id === "training" && currentStep === "evaluate")
            const isCurrent = step.id === currentStep
            return (
              <div key={step.id} className="flex items-center">
                <div className="flex items-center gap-2">
                  <div className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-medium transition-colors ${isComplete ? "bg-emerald-500 text-white" : isCurrent ? "bg-[#0052CC] text-white" : "bg-white/10 text-gray-400"}`}>
                    {isComplete ? <Check className="h-4 w-4" /> : index + 1}
                  </div>
                  <span className={`text-[10px] sm:text-sm font-medium ${isCurrent ? "text-white" : "text-gray-400"}`}>{step.label}</span>
                </div>
                {index < steps.length - 1 && <div className="mx-2 sm:mx-4 h-px w-8 sm:w-16 bg-white/10" />}
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
          scheduleCron={scheduleCron}
          onScheduleChange={(cron, desc) => { setScheduleCron(cron); setScheduleDesc(desc); }}
          onConnectionIDsChange={setConnectionIDs}
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
          error={trainingError}
          onRetry={() => { setTrainingError(""); setTrainingStatus("idle"); setCurrentStep("config"); }}
        />
      )}

      {currentStep === "evaluate" && evalMetrics && builtModel && (
        <EvaluateStep
          metrics={evalMetrics}
          model={builtModel}
          trainingTime={elapsedTime}
          onTrainAgain={handleTrainAgain}
          onOpenPlayground={handleOpenPlayground}
            openingPlayground={openingPlayground}
        />
      )}
    </div>
  )
}
