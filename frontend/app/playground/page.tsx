"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Sidebar } from "@/components/sidebar"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Checkbox } from "@/components/ui/checkbox"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from "@/components/ui/dialog"
import { Loader2, Play, Plus, FileSpreadsheet, Sparkles, CheckCircle2 } from "lucide-react"
import { api } from "@/lib/api"
import { useQueryStore } from "@/lib/query-store"

interface UploadedFile {
  file_id: string
  filename: string
  size: number
  created_at?: string
}

interface FineTunedModel {
  id: string
  name: string
  version: number
  source_file_id: string
  source_name: string
  model_path: string
  accuracy: number
  epochs: number
  batch_size: number
  created_at: string
}

const llmModels = [
  { id: "gpt-4o", name: "GPT-4o", provider: "OpenAI" },
  { id: "gpt-4o-mini", name: "GPT-4o Mini", provider: "OpenAI" },
  { id: "claude-sonnet-4-5", name: "Claude Sonnet 4.5", provider: "Anthropic" },
  { id: "claude-opus-4", name: "Claude Opus 4", provider: "Anthropic" },
]

export default function PlaygroundPage() {
  const router = useRouter()
  const { addQuery } = useQueryStore()
  
  const [models, setModels] = useState<any[]>([])
  const [fineTunedModels, setFineTunedModels] = useState<FineTunedModel[]>([])
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [selectedModel, setSelectedModel] = useState<string>("")
  const [currentModel, setCurrentModel] = useState<string>("")
  const [inputData, setInputData] = useState("[[1,2,3,4,5,6,7,8,9,10]]")
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  
  const [isCreateOpen, setIsCreateOpen] = useState(false)
  const [activeTab, setActiveTab] = useState("finetune")
  const [projectName, setProjectName] = useState("")
  const [selectedFiles, setSelectedFiles] = useState<string[]>([])
  const [selectedLLM, setSelectedLLM] = useState("gpt-4o")
  const [selectedExistingModel, setSelectedExistingModel] = useState<string | null>(null)
  
  // Training state
  const [isTraining, setIsTraining] = useState(false)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [trainingStatus, setTrainingStatus] = useState("")
  const [trainingEpoch, setTrainingEpoch] = useState(0)
  const [trainingTotalEpochs, setTrainingTotalEpochs] = useState(0)
  const [trainingLoss, setTrainingLoss] = useState(0)
  const [trainingAccuracy, setTrainingAccuracy] = useState(0)
  const [trainingComplete, setTrainingComplete] = useState(false)
  const [trainingETA, setTrainingETA] = useState("")

  useEffect(() => {
    loadModels()
    loadFineTunedModels()
    loadUploadedFiles()
  }, [])

  const loadModels = async () => {
    try {
      const data = await api.modelsList()
      setModels(data.models || [])
      setCurrentModel(data.current || "")
    } catch (error) {
      console.error("Failed to load models:", error)
    }
  }

  const loadFineTunedModels = async () => {
    try {
      const data = await api.getFineTunedModels()
      setFineTunedModels(data.models || [])
    } catch (error) {
      console.error("Failed to load fine-tuned models:", error)
    }
  }

  const loadUploadedFiles = async () => {
    try {
      const data = await api.getUploadedFiles()
      setUploadedFiles(data.files || [])
    } catch (error) {
      console.error("Failed to load files:", error)
    }
  }

  // Get files that haven't been trained yet
  const untrainedFiles = uploadedFiles.filter(file => 
    !fineTunedModels.some(model => model.source_file_id === file.file_id)
  )

  const handleModelSwitch = async () => {
    if (!selectedModel) return
    try {
      await api.modelsSwitch(selectedModel)
      setCurrentModel(selectedModel.split('/').pop()?.replace('.pt', '') || '')
    } catch (error) {
      console.error("Failed to switch model:", error)
    }
  }

  const handlePredict = async () => {
    setLoading(true)
    try {
      const values = JSON.parse(inputData)
      const data = await api.predict(values)
      setResult(data)
    } catch (error) {
      console.error("Prediction failed:", error)
    }
    setLoading(false)
  }

  const toggleFileSelection = (fileId: string) => {
    setSelectedFiles(prev => 
      prev.includes(fileId) ? prev.filter(x => x !== fileId) : [...prev, fileId]
    )
  }

  const formatFileSize = (bytes: number) => {
    if (!bytes) return ""
    if (bytes < 1024) return bytes + " B"
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB"
    return (bytes / (1024 * 1024)).toFixed(1) + " MB"
  }

  const handleFineTune = async () => {
    if (selectedFiles.length === 0) return
    
    setIsTraining(true)
    setTrainingProgress(0)
    setTrainingStatus("Starting training...")
    setTrainingComplete(false)
    setTrainingEpoch(0)
    setTrainingLoss(0)
    setTrainingAccuracy(0)
    
    try {
      const modelName = selectedFiles.length > 1 
        ? "multi_file_model" 
        : uploadedFiles.find(f => f.file_id === selectedFiles[0])?.filename || "model"
      
      // Start training (async)
      const trainPromise = api.multiTrain(selectedFiles, modelName, 15, 64, 0.001, 100)
      
      // Poll for progress
      let completed = false
      let lastEpoch = 0
      const startTime = Date.now()
      
      while (!completed) {
        await new Promise(r => setTimeout(r, 1500))
        
        try {
          const progress = await api.getTrainingProgress()
          
          if (progress.status === "training") {
            const epochs = progress.epochs || 15
            const epoch = progress.epoch || 0
            const epochProgress = epochs > 0 ? (epoch / epochs) * 100 : 0
            
            // Calculate ETA
            let eta = "Calculating..."
            if (epoch > 0 && epoch > lastEpoch) {
              const elapsed = (Date.now() - startTime) / 1000
              const timePerEpoch = elapsed / epoch
              const remaining = (epochs - epoch) * timePerEpoch
              if (remaining < 60) {
                eta = `${Math.round(remaining)}s remaining`
              } else if (remaining < 3600) {
                eta = `${Math.round(remaining / 60)}m remaining`
              } else {
                eta = `${(remaining / 3600).toFixed(1)}h remaining`
              }
            }
            lastEpoch = epoch
            
            setTrainingEpoch(epoch)
            setTrainingProgress(epochProgress)
            setTrainingLoss(progress.loss || 0)
            setTrainingAccuracy(progress.accuracy || 0)
            setTrainingStatus(`Epoch ${epoch}/${epochs} • ${eta}`)
          } else if (progress.status === "completed" || progress.status === "idle") {
            if (lastEpoch > 0) {
              completed = true
              setTrainingAccuracy(progress.accuracy || 0)
            }
          }
        } catch (e) {
          console.log("Progress poll:", e)
        }
      }
      
      await trainPromise
      
      setTrainingProgress(100)
      setTrainingComplete(true)
      setTrainingStatus("Training complete!")
      
      await loadFineTunedModels()
      
      await new Promise(r => setTimeout(r, 1500))
      setIsTraining(false)
      setSelectedFiles([])
      setActiveTab("models")
      
    } catch (error) {
      console.error("Training failed:", error)
      setTrainingStatus("Training failed!")
      setTimeout(() => setIsTraining(false), 2000)
    }
  }

  const handleCreateProject = async () => {
    if (!projectName.trim() || !selectedExistingModel) return
    
    try {
      const response = await api.createQuery(projectName, selectedLLM, [selectedExistingModel])
      addQuery({
        id: response.id,
        name: projectName,
        dataSources: [selectedExistingModel],
        model: selectedLLM,
      })
      resetDialog()
      router.push(`/playground/${response.id}`)
    } catch (error) {
      console.error("Failed to create project:", error)
    }
  }

  const resetDialog = () => {
    setIsCreateOpen(false)
    setProjectName("")
    setSelectedFiles([])
    setSelectedExistingModel(null)
    setIsTraining(false)
    setTrainingProgress(0)
    setTrainingComplete(false)
    setActiveTab("finetune")
  }

  return (
    <Sidebar>
      <div className="p-8 space-y-8 max-w-4xl">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold">Playground</h1>
            <p className="text-muted-foreground">Test your models with custom inputs</p>
          </div>
          
          <Dialog open={isCreateOpen} onOpenChange={(open) => { if (!open) resetDialog(); else setIsCreateOpen(true); }}>
            <DialogTrigger asChild>
              <Button>
                <Plus className="mr-2 h-4 w-4" />
                New
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-lg">
              <DialogHeader>
                <DialogTitle>Create New Project</DialogTitle>
                <DialogDescription>Fine-tune a model or select an existing one</DialogDescription>
              </DialogHeader>
              
              <Tabs value={activeTab} onValueChange={setActiveTab} className="mt-4">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="finetune">
                    <Sparkles className="h-4 w-4 mr-2" />
                    Fine-tune
                  </TabsTrigger>
                  <TabsTrigger value="models">
                    <FileSpreadsheet className="h-4 w-4 mr-2" />
                    Select Model
                  </TabsTrigger>
                </TabsList>
                
                {/* Tab 1: Fine-tune */}
                <TabsContent value="finetune" className="space-y-4 mt-4">
                  {!isTraining ? (
                    <>
                      <div className="space-y-2">
                        <Label>Select files to fine-tune</Label>
                        <div className="border rounded-lg max-h-60 overflow-y-auto">
                          {untrainedFiles.length === 0 ? (
                            <p className="text-sm text-muted-foreground p-4 text-center">
                              No files available. Upload files in Data Sources first.
                            </p>
                          ) : (
                            <div className="p-2 space-y-1">
                              {untrainedFiles.map((file) => (
                                <label
                                  key={file.file_id}
                                  className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-colors ${
                                    selectedFiles.includes(file.file_id) 
                                      ? "bg-primary/10 border border-primary/30" 
                                      : "hover:bg-secondary/50"
                                  }`}
                                >
                                  <Checkbox
                                    checked={selectedFiles.includes(file.file_id)}
                                    onCheckedChange={() => toggleFileSelection(file.file_id)}
                                  />
                                  <FileSpreadsheet className="h-4 w-4 text-green-500" />
                                  <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
                                    <span className="text-sm font-medium break-words line-clamp-2 leading-tight">{file.filename}</span>
                                    <span className="text-xs text-muted-foreground mt-0.5">{formatFileSize(file.size)}</span>
                                  </div>
                                </label>
                              ))}
                            </div>
                          )}
                        </div>
                        {selectedFiles.length > 0 && (
                          <p className="text-xs text-muted-foreground">{selectedFiles.length} file(s) selected</p>
                        )}
                      </div>
                      
                      <Button 
                        onClick={handleFineTune} 
                        disabled={selectedFiles.length === 0}
                        className="w-full"
                      >
                        <Sparkles className="mr-2 h-4 w-4" />
                        Start Fine-tuning
                      </Button>
                    </>
                  ) : (
                    <div className="py-6 space-y-6">
                      <div className="text-center">
                        {trainingComplete ? (
                          <CheckCircle2 className="h-12 w-12 text-green-500 mx-auto mb-4" />
                        ) : (
                          <Loader2 className="h-12 w-12 text-primary mx-auto mb-4 animate-spin" />
                        )}
                        <h3 className="text-lg font-semibold">
                          {trainingComplete ? "Training Complete!" : "Fine-tuning..."}
                        </h3>
                        <p className="text-sm text-muted-foreground mt-1">{trainingStatus}</p>
                      </div>
                      
                      <div className="space-y-4">
                        <div>
                          <div className="flex justify-between text-sm mb-2">
                            <span>Progress</span>
                            <span>{trainingProgress.toFixed(0)}%</span>
                          </div>
                          <Progress value={trainingProgress} className="h-2" />
                        </div>
                        
                        <div className="grid grid-cols-3 gap-3 text-center">
                          <div className="p-3 bg-secondary/50 rounded-lg">
                            <div className="text-xl font-bold">{trainingEpoch}/5</div>
                            <div className="text-xs text-muted-foreground">Epoch</div>
                          </div>
                          <div className="p-3 bg-secondary/50 rounded-lg">
                            <div className="text-xl font-bold">{trainingLoss.toFixed(3)}</div>
                            <div className="text-xs text-muted-foreground">Loss</div>
                          </div>
                          <div className="p-3 bg-secondary/50 rounded-lg">
                            <div className="text-xl font-bold text-green-600">{(trainingAccuracy * 100).toFixed(1)}%</div>
                            <div className="text-xs text-muted-foreground">Accuracy</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </TabsContent>
                
                {/* Tab 2: Select Model */}
                <TabsContent value="models" className="space-y-4 mt-4">
                  <div className="space-y-2">
                    <Label>Project Name</Label>
                    <Input
                      placeholder="e.g. Q4 Sales Analysis"
                      value={projectName}
                      onChange={(e) => setProjectName(e.target.value)}
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label>Select a trained model</Label>
                    <div className="border rounded-lg max-h-48 overflow-y-auto">
                      {fineTunedModels.length === 0 ? (
                        <p className="text-sm text-muted-foreground p-4 text-center">
                          No models available. Fine-tune a model first.
                        </p>
                      ) : (
                        <div className="p-2 space-y-1">
                          {fineTunedModels.map((model) => (
                            <label
                              key={model.id}
                              className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-colors ${
                                selectedExistingModel === model.id 
                                  ? "bg-primary/10 border border-primary/30" 
                                  : "hover:bg-secondary/50"
                              }`}
                              onClick={() => setSelectedExistingModel(model.id)}
                            >
                              <Checkbox
                                checked={selectedExistingModel === model.id}
                                onCheckedChange={() => setSelectedExistingModel(model.id)}
                              />
                              <Sparkles className="h-4 w-4 text-purple-500" />
                              <div className="flex flex-col flex-1 min-w-0">
                                <span className="text-sm font-medium truncate">{model.name}</span>
                                <span className="text-xs text-muted-foreground">
                                  v{model.version} · {(model.accuracy * 100).toFixed(1)}% accuracy
                                </span>
                              </div>
                            </label>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label>LLM Model</Label>
                    <Select value={selectedLLM} onValueChange={setSelectedLLM}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {llmModels.map((model) => (
                          <SelectItem key={model.id} value={model.id}>
                            {model.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <Button 
                    onClick={handleCreateProject} 
                    disabled={!projectName.trim() || !selectedExistingModel}
                    className="w-full"
                  >
                    Create Project
                  </Button>
                </TabsContent>
              </Tabs>
            </DialogContent>
          </Dialog>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Model Selection</CardTitle>
            <CardDescription>Current: {currentModel || "None"}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Select Model</Label>
              <Select onValueChange={setSelectedModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Choose a model" />
                </SelectTrigger>
                <SelectContent>
                  {models.map((model) => (
                    <SelectItem key={model.path} value={model.path}>
                      {model.name} {model.is_current && "(Current)"}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {selectedModel && (
              <Button onClick={handleModelSwitch}>
                Switch Model
              </Button>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Prediction</CardTitle>
            <CardDescription>Enter input data as JSON array</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Input Data</Label>
              <Textarea
                value={inputData}
                onChange={(e) => setInputData(e.target.value)}
                rows={4}
                placeholder='[[1,2,3,4,5,6,7,8,9,10]]'
              />
            </div>

            <Button onClick={handlePredict} disabled={loading}>
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Predicting...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Run Prediction
                </>
              )}
            </Button>

            {result && (
              <div className="mt-4 p-4 bg-slate-50 dark:bg-slate-900 border rounded">
                <h3 className="font-semibold mb-2">Results</h3>
                <div className="space-y-2 text-sm font-mono">
                  <div>
                    <span className="text-muted-foreground">Model:</span> {result.model_used}
                  </div>
                  <div>
                    <span className="text-muted-foreground">Predictions:</span> {JSON.stringify(result.predictions)}
                  </div>
                  <div>
                    <span className="text-muted-foreground">Confidences:</span> {result.confidences?.map((c: number) => c.toFixed(4)).join(', ')}
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </Sidebar>
  )
}
