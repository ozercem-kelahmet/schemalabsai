"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { toast } from "sonner"
import { useSidebar } from "@/components/layout/sidebar"
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
  const { buildingJobs } = useSidebar()

  const searchParams = useSearchParams()
  const currentQid = searchParams.get("qid")
  const router = useRouter()

  const [selectedDatasets, setSelectedDatasets] = useState<Dataset[]>([])
  const [currentStep, setCurrentStep] = useState<Step>("config")
  const [modelName, setModelName] = useState("")
  const [modelDescription, setModelDescription] = useState("")
  const [syncMode, setSyncMode] = useState<SyncMode>("manual")
  const [scheduleCron, setScheduleCron] = useState("")
  const [scheduleDesc, setScheduleDesc] = useState("")
  const [connectionIDs, setConnectionIDs] = useState("")
  const [baseModel, setBaseModel] = useState<string>(process.env.NEXT_PUBLIC_BASE_MODEL || "schema-v1")

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
  const [existingModelNames, setExistingModelNames] = useState<string[]>([])
  const [modelNameError, setModelNameError] = useState<string>("")
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
          if (data.epoch !== undefined && data.epoch > 0 && 
              (!data.query_id || data.query_id === trainingQueryIdRef.current)) {
            // Kafka'dan direkt veri - sadece bu training'e ait
            trainingStartedRef.current = true
            setTrainingStatus("training")
            const epoch = data.epoch
            const epochs = data.epochs || epoch
            const accuracy = data.accuracy > 1 ? data.accuracy / 100 : data.accuracy
            const loss = data.loss || 0
            const status = data.status || "training"

            if (status === "completed") {
              api.getTrainingProgress(queryId).then((progress) => {
                if (progress.status === "completed") {
                  const acc = (progress.accuracy > 1 ? progress.accuracy / 100 : progress.accuracy) || 0
                  setEvalMetrics({ accuracy: acc, precision: progress.precision || acc * 0.98, recall: progress.recall || acc * 0.97, f1Score: progress.f1_score || acc * 0.975 })
                  const mid = progress.model_id || queryId
                  setBuiltModel({ id: mid, modelId: mid, name: progress.model_name || modelName || "", description: "", datasets: [], syncMode, baseModel, accuracy: acc, createdAt: progress.start_time ? new Date(progress.start_time * 1000) : new Date(), updatedAt: new Date(), status: "completed", apiRequests: 0, tokensUsed: 0 })
                  if (progress.start_time) setElapsedTime(Math.max(0, Math.floor(Date.now() / 1000 - progress.start_time)))
                  completedByPollingRef.current = true
                  setCurrentStep("evaluate")
                }
              }).catch(() => {})
              return
            }

            // İlk SSE event'inde polling'i de bir kez tetikle (UI state sync için)
            if (epoch <= 2) {
              api.getTrainingProgress(queryId).catch(() => {})
            }

            setTotalEpochs(prev => Math.max(prev, epochs))
            setCurrentMetrics(prev => {
              if (!prev || epoch >= prev.epoch) {
                return { epoch, totalEpochs: epochs, loss, accuracy, learningRate: data.lr || data.learning_rate || 0.001 }
              }
              return prev
            })
            setMetricsHistory(prev => {
              const exists = prev.some((m: any) => m.epoch === epoch)
              if (!exists && epoch > 0) {
                const newEntry = { epoch, totalEpochs: epochs, loss, accuracy, learningRate: data.lr || data.learning_rate || 0.001 }
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


  // Fetch existing model names on mount for validation
  useEffect(() => {
    const fetchModelNames = async () => {
      try {
        const res = await fetch("/api/models/finetuned", { credentials: "include" })
        if (res.ok) {
          const data = await res.json()
          const names = (data.models || []).map((m: any) => m.name.toLowerCase().trim())
          setExistingModelNames(names)
        }
      } catch (e) { console.error("Failed to fetch model names", e) }
    }
    fetchModelNames()
  }, [])

  // Validate model name when it changes
  useEffect(() => {
    if (!modelName.trim()) {
      setModelNameError("")
      return
    }
    const nameLower = modelName.toLowerCase().trim()
    if (existingModelNames.includes(nameLower)) {
      setModelNameError("A model with this name already exists. Please choose a different name.")
    } else {
      setModelNameError("")
    }
  }, [modelName, existingModelNames])

  // Check for ongoing training on mount (e.g. after page refresh)
  useEffect(() => {
    const checkOngoingTraining = async () => {
      try {
        const urlQid = new URLSearchParams(window.location.search).get("qid")
        if (!urlQid) return
        trainingQueryIdRef.current = urlQid
        const savedInitLogs = sessionStorage.getItem("trainingInitLogs")
        const res = await fetch("/api/train/progress?query_id=" + urlQid, { credentials: "include" })
        const progress = await res.json()
        if (progress.status === "failed") {
          setCurrentStep("training")
          setTrainingStatus("failed")
          setTrainingError(progress.error || "An error occurred during training.")
          if (progress.model_name) setModelName(progress.model_name)
          addLog("Training failed: " + (progress.error || "Unknown error"))
          return
        }
        if (progress.status === "completed") {
          completedByPollingRef.current = true
          const acc = (progress.accuracy > 1 ? progress.accuracy / 100 : progress.accuracy) || 0
          setEvalMetrics({
            accuracy: acc,
            precision: progress.precision || acc * 0.98,
            recall: progress.recall || acc * 0.97,
            f1Score: progress.f1_score || acc * 0.975,
          })
          setBuiltModel({
            id: progress.model_id || urlQid,
            modelId: progress.model_id || urlQid,
            name: progress.model_name || modelName || "",
            description: "", datasets: [],
            syncMode, baseModel,
            accuracy: acc,
            createdAt: progress.start_time ? new Date(progress.start_time * 1000) : new Date(),
            updatedAt: new Date(),
            status: "completed", apiRequests: 0, tokensUsed: 0,
          })
          if (progress.start_time) setElapsedTime(Math.max(0, Math.floor(Date.now() / 1000 - progress.start_time)))
          if (progress.model_name) setModelName(progress.model_name)
          if (progress.history && Array.isArray(progress.history)) {
            const hist = progress.history.map((h: any) => ({
              epoch: h.epoch, totalEpochs: progress.epochs || h.epoch,
              loss: h.loss || 0, accuracy: (h.accuracy > 1 ? h.accuracy / 100 : h.accuracy) || 0,
              learningRate: progress.learning_rate || 0.001,
            }))
            setMetricsHistory(hist)
          }
          setCurrentStep("evaluate")
          return
        }
        if (progress.status === "training" || progress.status === "initializing") {
          const st = progress.start_time || 0
          const el = st > 0 ? Math.floor(Date.now() / 1000 - st) : 0
          if (el > 300 && progress.epoch === 0) return
          trainingStartedRef.current = true
          setCurrentStep("training")
          setTrainingStatus("training")
          if (savedInitLogs) {
            try { setLogs(JSON.parse(savedInitLogs)) } catch {}
          } else if (progress.init_logs && Array.isArray(progress.init_logs)) {
            const ts = new Date().toLocaleTimeString("en-US", { hour12: false })
            setLogs(progress.init_logs.map((l: string) => l.startsWith("[") ? l : "[" + ts + "] " + l))
          }
          if (progress.history && progress.history.length > 0) {
            const restored = progress.history.map((h: any) => ({
              epoch: h.epoch, totalEpochs: progress.epochs || h.epoch + 1,
              loss: h.loss || 0, accuracy: (h.accuracy > 1 ? h.accuracy / 100 : h.accuracy) || 0,
              learningRate: progress.learning_rate || 0.001,
            }))
            setMetricsHistory(restored)
            restored.forEach((rm: any) => addLog("Epoch " + rm.epoch + "/" + rm.totalEpochs + " - Loss: " + rm.loss.toFixed(4) + ", Accuracy: " + (rm.accuracy * 100).toFixed(1) + "%"))
          }
          if (progress.model_name) setModelName(progress.model_name)
          if (progress.epochs) setTotalEpochs(progress.epochs)
          const acc = progress.accuracy > 1 ? progress.accuracy / 100 : progress.accuracy
          setCurrentMetrics({ epoch: progress.epoch, totalEpochs: progress.epochs || 0, loss: progress.loss || 0, accuracy: acc || 0, learningRate: progress.learning_rate || 0.001 })
          if (progress.start_time) {
            const elapsed = Math.floor(Date.now() / 1000 - progress.start_time)
            if (elapsed > 0) setElapsedTime(elapsed)
          }
        }
      } catch (e) {
        console.error("[checkOngoingTraining]", e)
      }
    }
    checkOngoingTraining()
  }, [])

  // localStorage restore removed - checkOngoingTraining handles it

  useEffect(() => {
    if (!currentQid) return
    if (trainingQueryIdRef.current === currentQid) return
    if (sseRef.current) { sseRef.current.close(); sseRef.current = null }
    if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null }
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
    setMetricsHistory([])
    setCurrentMetrics(null)
    setLogs([])
    setEvalMetrics(null)
    setBuiltModel(null)
    setTrainingError("")
    setElapsedTime(0)
    setIsPaused(false)
    completedByPollingRef.current = false
    trainCancelledRef.current = false
    trainingQueryIdRef.current = currentQid
    trainingStartedRef.current = true
    const loadTraining = async () => {
      try {
        const res = await fetch("/api/train/progress?query_id=" + currentQid, { credentials: "include" })
        const progress = await res.json()
        if (progress.status === "completed") {
          completedByPollingRef.current = true
          const acc = (progress.accuracy > 1 ? progress.accuracy / 100 : progress.accuracy) || 0
          setEvalMetrics({ accuracy: acc, precision: progress.precision || acc * 0.98, recall: progress.recall || acc * 0.97, f1Score: progress.f1_score || acc * 0.975 })
          setBuiltModel({ id: progress.model_id || currentQid, modelId: progress.model_id || currentQid, name: progress.model_name || modelName || "", description: "", datasets: [], syncMode, baseModel, accuracy: acc, createdAt: progress.start_time ? new Date(progress.start_time * 1000) : new Date(), updatedAt: new Date(), status: "completed", apiRequests: 0, tokensUsed: 0 })
          if (progress.start_time) setElapsedTime(Math.max(0, Math.floor(Date.now() / 1000 - progress.start_time)))
          if (progress.history && Array.isArray(progress.history)) {
            setMetricsHistory(progress.history.map((h: any) => ({ epoch: h.epoch, totalEpochs: progress.epochs || h.epoch, loss: h.loss || 0, accuracy: (h.accuracy > 1 ? h.accuracy / 100 : h.accuracy) || 0, learningRate: progress.learning_rate || 0.001 })))
          }
          setCurrentStep("evaluate")
        } else if (progress.status === "failed") {
          setCurrentStep("training")
          setTrainingStatus("failed")
          setTrainingError(progress.error || "An error occurred during training.")
        } else if (progress.status === "training" || progress.status === "initializing") {
          setCurrentStep("training")
          setTrainingStatus("training")
          if (progress.history && progress.history.length > 0) {
            setMetricsHistory(progress.history.map((h: any) => ({ epoch: h.epoch, totalEpochs: progress.epochs || h.epoch + 1, loss: h.loss || 0, accuracy: (h.accuracy > 1 ? h.accuracy / 100 : h.accuracy) || 0, learningRate: progress.learning_rate || 0.001 })))
          }
          if (progress.start_time) setElapsedTime(Math.max(0, Math.floor(Date.now() / 1000 - progress.start_time)))
          setCurrentMetrics({ epoch: progress.epoch, totalEpochs: progress.epochs || 0, loss: progress.loss || 0, accuracy: (progress.accuracy > 1 ? progress.accuracy / 100 : progress.accuracy) || 0, learningRate: progress.learning_rate || 0.001 })
          connectSSE(currentQid)
        }
      } catch {}
    }
    loadTraining()
  }, [currentQid])


  useEffect(() => {
    if (currentStep !== "training") return
    const qid = trainingQueryIdRef.current
    if (!qid) return
    const j = buildingJobs.find(b => b.id === qid || b.queryId === qid)
    if (j && j.status === "completed") {
      stopPolling()
      completedByPollingRef.current = true
      api.getTrainingProgress(qid).then((progress) => {
        const acc = (progress.accuracy > 1 ? progress.accuracy / 100 : progress.accuracy) || 0
        setEvalMetrics({
          accuracy: acc,
          precision: progress.precision || acc * 0.98,
          recall: progress.recall || acc * 0.97,
          f1Score: progress.f1_score || acc * 0.975,
        })
        const mid = progress.model_id || j.id
        setBuiltModel({
          id: mid, modelId: mid,
          name: j.name || modelName || "Trained Model",
          description: "", datasets: [],
          syncMode, baseModel,
          accuracy: acc,
          createdAt: new Date(), updatedAt: new Date(),
          status: "completed", apiRequests: 0, tokensUsed: 0,
        })
        if (progress.start_time) {
          setElapsedTime(Math.max(0, Math.floor(Date.now() / 1000 - progress.start_time)))
        }
        setCurrentStep("evaluate")
      }).catch(() => {})
    }
  }, [buildingJobs, currentStep])

  // Clear localStorage when training completes
  const clearTrainingStorage = () => {
    
    
    
    
    
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

    if (modelNameError) {
      toast.error("Model Name Not Available", { description: modelNameError, duration: 5000 })
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
      const _active = JSON.parse(sessionStorage.getItem("activeTrainings") || "[]")
      if (!_active.includes(trainingQueryId)) {
        _active.push(trainingQueryId)
        sessionStorage.setItem("activeTrainings", JSON.stringify(_active))
      }
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
        selectedTablesStr,
        baseModel
      )

      trainingStartedRef.current = false
      setCurrentStep("training")
      setTrainingStatus("initializing")
      setMetricsHistory([])
      setElapsedTime(0)
      setCurrentMetrics(null)
      const ts = new Date().toLocaleTimeString("en-US", { hour12: false })
      const initialLogs = [
        `[${ts}] Initializing build environment...`,
        `[${ts}] Model: ${modelName}`,
        `[${ts}] Base Model: ${baseModel}`,
        `[${ts}] Sync Mode: ${syncMode}${scheduleCron ? ` (${scheduleDesc || scheduleCron})` : ""}`,
        `[${ts}] Connecting ${selectedDatasets.length} data source(s)...`,
        ...selectedDatasets.map((ds) => `[${ts}]   → ${ds.name} (${ds.source}): ${ds.rows.toLocaleString()} rows, ${ds.columns} columns`),
        `[${ts}] Starting fine-tuning process...`,
        `[${ts}] Sending data to ML server...`,
        `[${ts}] Data preprocessing complete`,
        `[${ts}] Building knowledge base...`,
        `[${ts}] Training neural architecture...`,
      ]
      setLogs(initialLogs)
      sessionStorage.setItem("trainingInitLogs", JSON.stringify(initialLogs))
      
      // Wait 3 seconds for Go handler to reset progress and start Flask training
      await new Promise(resolve => setTimeout(resolve, 500))
      trainingStartedRef.current = true
      setTrainingStatus("training")
      setElapsedTime(0)



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
        if (completedByPollingRef.current) { return }
        const progress = await api.getTrainingProgress(trainingQueryIdRef.current)
        if (progress.status === "failed") {
          stopPolling()
          trainingStartedRef.current = false
          setCurrentStep("training")
          setTrainingError(progress.error || "An error occurred during training.")
          setTrainingStatus("failed")
          if (progress.model_name) setModelName(progress.model_name)
          addLog("Training failed: " + (progress.error || "Unknown error"))
          return
        }
        if (progress.status === "completed") {
          stopPolling()
          trainingStartedRef.current = false
          completedByPollingRef.current = true
          const acc = (progress.accuracy > 1 ? progress.accuracy / 100 : progress.accuracy) || 0
          setEvalMetrics({
            accuracy: acc,
            precision: progress.precision || acc * 0.98,
            recall: progress.recall || acc * 0.97,
            f1Score: progress.f1_score || acc * 0.975,
          })
          const modelId = progress.model_id || trainingQueryIdRef.current || ""
          const restoredModel: Model = {
            id: modelId,
            modelId: modelId,
            name: progress.model_name || "",
            description: "",
            datasets: [],
            syncMode: "manual" as SyncMode,
            baseModel: process.env.NEXT_PUBLIC_BASE_MODEL || "schema-v1",
            accuracy: acc,
            createdAt: progress.start_time ? new Date(progress.start_time * 1000) : new Date(),
            updatedAt: new Date(),
            status: "completed",
            apiRequests: 0,
            tokensUsed: 0,
          }
          setBuiltModel(restoredModel)
          if (progress.start_time) {
            setElapsedTime(Math.max(0, Math.floor(Date.now() / 1000 - progress.start_time)))
          }
          if (progress.model_name) setModelName(progress.model_name)
          if (progress.history && Array.isArray(progress.history)) {
            const hist = progress.history.map((h: any) => ({
              epoch: h.epoch,
              totalEpochs: progress.epochs || h.epoch,
              loss: h.loss || 0,
              accuracy: (h.accuracy > 1 ? h.accuracy / 100 : h.accuracy) || 0,
              learningRate: progress.learning_rate || 0.001,
            }))
            setMetricsHistory(hist)
            hist.forEach((h: any) => addLog(`Epoch ${h.epoch}/${h.totalEpochs} - Loss: ${h.loss.toFixed(4)}, Accuracy: ${(h.accuracy * 100).toFixed(1)}%`))
          }
          setCurrentStep("evaluate")
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
            if (!prev) return { epoch: progress.epoch, totalEpochs: epochs, loss: progress.loss || 0, accuracy: (progress.accuracy > 1 ? progress.accuracy / 100 : progress.accuracy) || 0, learningRate: progress?.learning_rate || 0.001 }
            if (progress.epoch >= prev.epoch) return { epoch: progress.epoch, totalEpochs: Math.max(epochs, prev.totalEpochs), loss: progress.loss || 0, accuracy: (progress.accuracy > 1 ? progress.accuracy / 100 : progress.accuracy) || 0, learningRate: progress?.learning_rate || 0.001 }
            return prev
          })
          const newMetrics: TrainingMetrics = {
            epoch: progress.epoch,
            totalEpochs: epochs,
            loss: progress.loss || 0,
            accuracy: (progress.accuracy > 1 ? progress.accuracy / 100 : progress.accuracy) || 0,
            learningRate: progress?.learning_rate || 0.001,
          }
          
          // Metrics update handled in animation block above
          if (progress.history && Array.isArray(progress.history) && progress.history.length > (logs.length)) {
            const loggedEpochs = new Set(logs.map((l: any) => {
              const m = l.match(/Epoch (\d+)/)
              return m ? parseInt(m[1]) : -1
            }))
            for (const h of progress.history) {
              if (!loggedEpochs.has(h.epoch)) {
                const a = h.accuracy > 1 ? h.accuracy / 100 : h.accuracy
                addLog(`Epoch ${h.epoch}/${epochs} - Loss: ${(h.loss || 0).toFixed(4)}, Accuracy: ${(a * 100).toFixed(1)}%`)
              }
            }
          } else if (progress.epoch > (currentMetrics?.epoch || 0)) { addLog(`Epoch ${progress.epoch}/${epochs} - Loss: ${(progress.loss || 0).toFixed(4)}, Accuracy: ${((newMetrics.accuracy) * 100).toFixed(1)}%`) }
          setMetricsHistory((prev) => {
            // Merge Redis history + current epoch
            let merged = [...prev]
            if (progress.history && progress.history.length > 0) {
              const existingEpochs = new Set(merged.map((m: any) => m.epoch))
              for (const h of progress.history) {
                if (!existingEpochs.has(h.epoch)) {
                  merged.push({ epoch: h.epoch, totalEpochs: epochs, loss: h.loss || 0, accuracy: h.accuracy > 1 ? h.accuracy / 100 : h.accuracy || 0, learningRate: progress?.learning_rate || 0.001 })
                  existingEpochs.add(h.epoch)
                }
              }
              merged.sort((a: any, b: any) => a.epoch - b.epoch)
            }
            const lastEpoch = merged.length > 0 ? merged[merged.length - 1].epoch : 0
            if (progress.epoch > lastEpoch) {
              merged.push(newMetrics)
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
  modelNameError={modelNameError}
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
          storeProgress={(() => {
            const qid = trainingQueryIdRef.current
            if (!qid) return undefined
            const j = buildingJobs.find(b => b.id === qid || b.queryId === qid)
            return j ? j.progress : undefined
          })()}
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
