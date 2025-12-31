"use client"

import type React from "react"
import { Dialog, DialogTrigger, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogClose } from "@/components/ui/dialog"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { cn } from "@/lib/utils"
import {
  Shield,
  LayoutGrid,
  Database,
  Plus,
  Key,
  Globe,
  Settings,
  Pencil,
  Trash2,
  ChevronRight,
  Folder,
  FileSpreadsheet,
  Server,
  Copy,
  PanelLeft,
  MessageSquare,
  X,
  Sparkles,
  Loader2,
  CheckCircle2,
  Table2,
} from "lucide-react"
import { useState, useRef, useEffect } from "react"
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts"
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuSeparator, ContextMenuTrigger } from "@/components/ui/context-menu"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Checkbox } from "@/components/ui/checkbox"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Tooltip as UITooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Progress } from "@/components/ui/progress"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { SchemaProcessingAnimation } from "@/components/schema-processing-animation"
import { useQueryStore, type QueryItem } from "@/lib/query-store"
import { useAuth } from "@/lib/auth"
import { api } from "@/lib/api"
import {
  Sidebar as SidebarPrimitive,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar"
import { TeamSwitcher } from "@/components/team-switcher"
import { NavUser } from "@/components/nav-user"


const teams = [
  { name: "Schema Labs", logo: () => <span className="font-bold">SL</span>, plan: "Pro" },
  { name: "Personal", logo: () => <span className="font-bold">P</span>, plan: "Free" },
  { name: "Acme Corp", logo: () => <span className="font-bold">AC</span>, plan: "Enterprise" },
]

interface UploadedFile {
  file_id: string
  filename: string
  path: string
  size: number
  folder_id?: string | null
  folder_name?: string | null
}

interface FineTunedModel {
  id: string
  name: string
  source_file_id: string
  accuracy: number
  version: number
  created_at: string
  epochs?: number
  batch_size?: number
  loss?: number
}

function SidebarInner() {
  const { user, logout } = useAuth()
  const pathname = usePathname()
  const router = useRouter()
  const { state } = useSidebar()

  const { queries, addQuery, updateQuery, deleteQuery, duplicateQuery } = useQueryStore()

  const [editingChatId, setEditingChatId] = useState<string | null>(null)
  const [editingName, setEditingName] = useState("")
  const [isNewChatOpen, setIsNewChatOpen] = useState(false)
  const [activeTab, setActiveTab] = useState("finetune")
  const [newProjectName, setNewProjectName] = useState("")
  const [selectedModel, setSelectedModel] = useState("gpt-4o")
  const [selectedFiles, setSelectedFiles] = useState<string[]>([])
  const [selectedExistingModel, setSelectedExistingModel] = useState<string | null>(null)
  const [editingModelId, setEditingModelId] = useState<string | null>(null)
  const [editingModelName, setEditingModelName] = useState("")
  const [expandedFolders, setExpandedFolders] = useState<string[]>(["files"])
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingProjectName, setProcessingProjectName] = useState("")
  const [processingDataSources, setProcessingDataSources] = useState<string[]>([])
  const pendingNavigationRef = useRef<string | null>(null)
  
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [fineTunedModels, setFineTunedModels] = useState<FineTunedModel[]>([])
  const [connections, setConnections] = useState<any[]>([])
  
  const [dbTables, setDbTables] = useState<{[key: string]: string[]}>({})
  const [isTraining, setIsTraining] = useState(false)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [trainingStatus, setTrainingStatus] = useState("")
  const [trainingEpoch, setTrainingEpoch] = useState(0)
  
  // Query-based training states for multiple simultaneous trainings
  const [trainingStates, setTrainingStates] = useState<Record<string, {
    epoch: number
    epochs: number
    accuracy: number
    loss: number
    status: string
    eta: string
    progress: number
  }>>({})
  const [epochCount, setEpochCount] = useState(5)
  const [displayEpochs, setDisplayEpochs] = useState(5)
  const [batchSize, setBatchSize] = useState(32)
  const [trainingLoss, setTrainingLoss] = useState(0)
  const [lossHistory, setLossHistory] = useState<{epoch: number, loss: number, accuracy: number}[]>([])
  const [trainingAccuracy, setTrainingAccuracy] = useState(0)
  const [trainingETA, setTrainingETA] = useState("")
  const trainingStartTimeRef = useRef(0)
  const [trainingComplete, setTrainingComplete] = useState(false)
  const [mergedCsvPath, setMergedCsvPath] = useState<string | null>(null)
  const [trainingQueryId, setTrainingQueryId] = useState<string | null>(null)
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)

  const startPolling = () => {
    if (pollIntervalRef.current) return
    pollIntervalRef.current = setInterval(async () => {
      try {
        // Poll for current modal training
        if (trainingQueryId) {
          const progress = await api.getTrainingProgress(trainingQueryId)
          if (progress.status === "training") {
            setTrainingEpoch(progress.epoch)
            setTrainingAccuracy(progress.accuracy / 100)
            setTrainingLoss(progress.loss)
            setDisplayEpochs(progress.epochs)
            setLossHistory(prev => {
              if (prev.length === 0 || prev[prev.length-1].epoch !== progress.epoch) {
                return [...prev, {epoch: progress.epoch, loss: progress.loss, accuracy: progress.accuracy}]
              }
              return prev
            })
            setTrainingProgress(20 + (progress.epoch / progress.epochs) * 70)
            if (progress.eta) {
              setTrainingETA(progress.eta)
            }
            setTrainingStatus("Epoch " + progress.epoch + "/" + progress.epochs)
            
            // Update trainingStates for this query
            setTrainingStates(prev => ({
              ...prev,
              [trainingQueryId]: {
                epoch: progress.epoch,
                epochs: progress.epochs,
                accuracy: progress.accuracy,
                loss: progress.loss,
                status: progress.status,
                eta: progress.eta || "",
                progress: 20 + (progress.epoch / progress.epochs) * 70
              }
            }))
          } else if (progress.status === "completed" && progress.epoch > 0 && progress.epoch === progress.epochs) {
            setTrainingProgress(100)
            setTrainingComplete(true)
            setTrainingStatus("Done! Accuracy: " + progress.accuracy.toFixed(1) + "%")
            stopPolling()
            updateQuery(trainingQueryId, { isTraining: false, hasModel: true, trainingModelId: progress.model_id })
            setTrainingStates(prev => {
              const newState = { ...prev }
              delete newState[trainingQueryId]
              return newState
            })
          }
        }
        
        // Also poll for all training queries in sidebar
        const trainingQueries = queries.filter(q => q.isTraining && q.id !== trainingQueryId)
        for (const query of trainingQueries) {
          const progress = await api.getTrainingProgress(query.id)
          if (progress.status === "training") {
            setTrainingStates(prev => ({
              ...prev,
              [query.id]: {
                epoch: progress.epoch,
                epochs: progress.epochs,
                accuracy: progress.accuracy,
                loss: progress.loss,
                status: progress.status,
                eta: progress.eta || "",
                progress: 20 + (progress.epoch / progress.epochs) * 70
              }
            }))
          } else if (progress.status === "completed" && progress.epoch > 0) {
            updateQuery(query.id, { isTraining: false, hasModel: true, trainingModelId: progress.model_id })
            setTrainingStates(prev => {
              const newState = { ...prev }
              delete newState[query.id]
              return newState
            })
          }
        }
      } catch (e) {}
    }, 1000)
  }

  const stopPolling = () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current)
      pollIntervalRef.current = null
    }
  }

  useEffect(() => {
    if (isNewChatOpen) {
      // Always reset for new modal open
      stopPolling()
      // Reset training state for new project
      setTrainingProgress(0)
      setTrainingEpoch(0)
      setTrainingLoss(0)
      setTrainingAccuracy(0)
      setTrainingStatus("")
      setTrainingComplete(false)
      setTrainingQueryId(null)
      setLossHistory([])
      setTrainingETA("")
      setIsTraining(false)
      loadData()
    }
  }, [isNewChatOpen])

  // Start polling for training queries on mount
  useEffect(() => {
    const trainingQueries = queries.filter(q => q.isTraining)
    if (trainingQueries.length > 0) {
      startPolling()
    }
    return () => stopPolling()
  }, [queries])

  const loadData = async () => {
    try {
      const [filesData, modelsData, connectionsData] = await Promise.all([
        api.getUploadedFiles(),
        api.getFineTunedModels(),
        api.getConnections()
      ])
      setUploadedFiles(filesData.files || [])
      setFineTunedModels(modelsData.models || [])
      setConnections(connectionsData.connections || [])
    } catch (error) {
      console.error("Failed to load data:", error)
    }
  }

  const startEditing = (query: QueryItem) => {
    setEditingChatId(query.id)
    setEditingName(query.name)
  }

  const saveEdit = () => {
    if (editingChatId && editingName.trim()) {
      updateQuery(editingChatId, { name: editingName.trim() })
    }
    setEditingChatId(null)
  }

  const handleDeleteQuery = (queryId: string) => {
    deleteQuery(queryId)
    if (pathname === "/playground/" + queryId) {
      router.push("/")
    }
  }

 
  const handleDuplicateQuery = async (query: QueryItem) => {
    const duplicated = await duplicateQuery(query.id)
    if (duplicated) router.push("/playground/" + duplicated.id)
  }

  const toggleFileSelection = async (fileId: string) => {
    setSelectedFiles(prev => prev.includes(fileId) ? prev.filter(id => id !== fileId) : [...prev, fileId])
    
    // If it's a connection, load tables
    const conn = connections.find(c => c.id === fileId)
    if (conn && !dbTables[fileId]) {
      try {
        const result = await api.listTables(fileId)
        if (result.tables) {
          setDbTables(prev => ({ ...prev, [fileId]: result.tables }))
        }
      } catch (error) {
        console.error("Failed to load tables:", error)
      }
    }
  }

  const toggleFolder = (folderId: string) => {
    setExpandedFolders(prev => prev.includes(folderId) ? prev.filter(id => id !== folderId) : [...prev, folderId])
  }

  const formatFileSize = (bytes: number) => {
    if (!bytes) return ""
    if (bytes < 1024) return bytes + " B"
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB"
    return (bytes / (1024 * 1024)).toFixed(1) + " MB"
  }

  const handleFineTune = async () => {
    if (selectedFiles.length === 0 || !newProjectName.trim()) return
    
    const projectName = newProjectName.trim()
    const newQuery = await addQuery({
      name: projectName,
      model: selectedModel,
      dataSources: selectedFiles,
      isTraining: true,
      hasModel: false
    })
    setTrainingQueryId(newQuery.id)

    setIsTraining(true)
    setTrainingProgress(5)
    setTrainingStatus("Analyzing data...")
    trainingStartTimeRef.current = Date.now()
    setTrainingETA("")
      setIsTraining(false)
    setTrainingComplete(false)
    setTrainingEpoch(0)
    setTrainingLoss(0)
    setLossHistory([])
    setTrainingAccuracy(0)
    
    try {
      // Export tables to CSV first if any database tables selected
      const tableSelections = selectedFiles.filter(id => id.includes(':'))
      const fileSelections = selectedFiles.filter(id => !id.includes(':'))
      
      let exportedFileIds: string[] = []
      if (tableSelections.length > 0) {
        setTrainingStatus("Exporting database tables...")
        for (const tableId of tableSelections) {
          const [connId, tableName] = tableId.split(':')
          const result = await api.exportTable(connId, tableName)
          if (result.file_id) {
            exportedFileIds.push(result.file_id)
          }
        }
      }
      
      const allFileIds = [...fileSelections, ...exportedFileIds]
      
      const analysis = await api.analyzeFiles(allFileIds)
      const smartEpochs = analysis.smart_epochs || epochCount
      const smartBatch = analysis.smart_batch_size || batchSize
      setDisplayEpochs(smartEpochs)
      
      setTrainingStatus("Training " + (analysis.n_samples || 0) + " samples, " + (analysis.n_classes || 0) + " classes, " + smartEpochs + " epochs...")
      setTrainingProgress(10)
      
      const fileNames = selectedFiles.map(id => {
        const file = uploadedFiles.find(f => f.file_id === id)
        return file?.filename?.replace(/\.[^/.]+$/, "") || ""
      }).filter(Boolean)
      const modelName = fileNames.length > 1
        ? "model_merged_" + fileNames.length + "files_" + new Date().toISOString().slice(0,10).replace(/-/g, "")
        : "model_" + fileNames[0] + "_" + new Date().toISOString().slice(0,10).replace(/-/g, "")
      
      startPolling()
      setTrainingProgress(20)
      const result = await api.multiTrain(allFileIds, modelName, smartEpochs, smartBatch, 0.001, 100, newQuery.id)
      stopPolling()
      
      if (result.accuracy) setTrainingAccuracy(result.accuracy / 100)
      if (result.loss) setTrainingLoss(result.loss)
      setTrainingEpoch(result.epochs || smartEpochs)
      
      if (result.merged_csv) setMergedCsvPath(result.merged_csv)
      setTrainingProgress(100)
      setTrainingComplete(true)
      setTrainingStatus("Done! " + (result.files_merged || 1) + " files, " + (result.rows || 0) + " rows, " + (result.epochs || smartEpochs) + " epochs, " + (result.accuracy || 0).toFixed(1) + "% acc")
      
      updateQuery(newQuery.id, { isTraining: false, hasModel: true, trainingModelId: result.model_id })
      
      await loadData()
      await new Promise(r => setTimeout(r, 2000))
      setIsTraining(false)
      setSelectedFiles([])
      setNewProjectName("")
      setTrainingQueryId(null)
      setIsNewChatOpen(false)
      router.push("/playground/" + newQuery.id)
      
    } catch (error) {
      console.error("Training failed:", error)
      setTrainingStatus("Training failed!")
      // Query silme kaldırıldı - training hata verse de kayıt kalsın
      setTimeout(() => {
        setIsTraining(false)
        setTrainingQueryId(null)
      }, 2000)
    }
  }

  const handleCreateProject = async () => {
    if (!newProjectName.trim() || !selectedExistingModel) return
    
    const projectName = newProjectName.trim()
    
    setProcessingProjectName(projectName)
    const model = fineTunedModels.find(m => m.id === selectedExistingModel)
    setProcessingDataSources([model?.name || "Model"])
    setIsProcessing(true)
    setIsNewChatOpen(false)

    const newQuery = await addQuery({
      name: projectName,
      model: selectedModel,
      dataSources: [selectedExistingModel],
      hasModel: true
    })

    pendingNavigationRef.current = newQuery.id
    resetDialog()

    setTimeout(() => {
      setIsProcessing(false)
      setProcessingProjectName("")
      setProcessingDataSources([])
      if (pendingNavigationRef.current) {
        router.push("/playground/" + pendingNavigationRef.current)
        pendingNavigationRef.current = null
      }
    }, 3000)
  }

  const resetDialog = () => {
    setNewProjectName("")
    setSelectedModel("gpt-4o")
    setSelectedFiles([])
    setSelectedExistingModel(null)
    setIsTraining(false)
    setTrainingProgress(0)
    setTrainingComplete(false)
    setActiveTab("finetune")
    setTrainingQueryId(null)
  }

  const handleDeleteModel = async (modelId: string) => {
    try {
      await api.deleteFineTunedModel(modelId)
      setFineTunedModels(prev => prev.filter(m => m.id !== modelId))
      if (selectedExistingModel === modelId) setSelectedExistingModel(null)
    } catch (error) {
      console.error("Failed to delete model:", error)
    }
  }

  const handleRenameModel = async (modelId: string) => {
    if (!editingModelName.trim()) return
    try {
      await api.renameFineTunedModel(modelId, editingModelName.trim())
      setFineTunedModels(prev => prev.map(m => m.id === modelId ? {...m, name: editingModelName.trim()} : m))
      setEditingModelId(null)
      setEditingModelName("")
    } catch (error) {
      console.error("Failed to rename model:", error)
    }
  }

  const handleCloseDialog = () => {
    if (isTraining) {
      setIsNewChatOpen(false)
    } else {
      resetDialog()
      setIsNewChatOpen(false)
    }
  }

  const handleQueryClick = (query: QueryItem) => {
    // Always navigate to playground - training view will be shown there if isTraining is true
    router.push("/playground/" + query.id)
  }

  const navItems = [
    { icon: LayoutGrid, label: "Overview", href: "/" },
    { icon: Database, label: "Data Sources", href: "/data-sources" },
  ]

  const configItems = [
    { icon: Key, label: "API Keys", href: "/api-keys" },
    { icon: Globe, label: "Endpoints", href: "/endpoints" },
    { icon: Settings, label: "Settings", href: "/settings" },
    { icon: Shield, label: "Admin", href: "/admin", adminOnly: true },
  ]

  const recentQueries = queries.slice(0, 10)
  
  const allFilesSelected = uploadedFiles.length > 0 && uploadedFiles.every(f => selectedFiles.includes(f.file_id))
  const someFilesSelected = selectedFiles.length > 0 && !allFilesSelected

  return (
    <>
      <SidebarPrimitive collapsible="icon">
        <SidebarHeader>
          <TeamSwitcher teams={teams} />
        </SidebarHeader>

        <SidebarContent>
          <SidebarGroup>
            <SidebarMenu>
              {navItems.map((item) => (
                <SidebarMenuItem key={item.href}>
                  <SidebarMenuButton asChild isActive={pathname === item.href} tooltip={item.label}>
                    <Link href={item.href}>
                      <item.icon />
                      <span>{item.label}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroup>

          <SidebarGroup className="flex flex-col min-h-0">
            <SidebarGroupLabel>Playground</SidebarGroupLabel>
            <SidebarMenu>
              <SidebarMenuItem>
                <Dialog open={isNewChatOpen} onOpenChange={setIsNewChatOpen}>
                  <DialogTrigger asChild>
                    <SidebarMenuButton tooltip="New">
                      <Plus />
                      <span>New</span>
                    </SidebarMenuButton>
                  </DialogTrigger>
                  <DialogContent className="sm:max-w-lg max-h-[90vh] overflow-y-auto" onInteractOutside={(e) => e.preventDefault()} onEscapeKeyDown={(e) => e.preventDefault()}>
                    <DialogHeader>
                      <DialogTitle>Create New Project</DialogTitle>
                      <DialogDescription>Fine-tune a model or select an existing one</DialogDescription>
                    </DialogHeader>
                    <button onClick={handleCloseDialog} className="absolute right-4 top-4 z-50 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none"><X className="h-4 w-4" /></button>
                    
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
                      
                      <TabsContent value="finetune" className="space-y-4 mt-4">
                        {!isTraining ? (
                          <>
                            <div className="space-y-2">
                              <Label>Project Name</Label>
                              <Input placeholder="e.g. Q4 Sales Analysis" value={newProjectName} onChange={(e) => setNewProjectName(e.target.value)} className="bg-secondary" />
                            </div>
                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <Label>Select Data Sources</Label>
                                <span className="text-xs text-muted-foreground">{selectedFiles.length} selected</span>
                              </div>
                              
                              <div className="rounded-lg border border-border bg-secondary/30 max-h-[280px] overflow-y-auto">
                                {/* Group files by folder */}
                                {(() => {
                                  const filesByFolder: Record<string, UploadedFile[]> = {}
                                  const rootFiles: UploadedFile[] = []
                                  
                                  uploadedFiles.forEach(file => {
                                    if (file.folder_id && file.folder_name) {
                                      if (!filesByFolder[file.folder_id]) {
                                        filesByFolder[file.folder_id] = []
                                      }
                                      filesByFolder[file.folder_id].push(file)
                                    } else {
                                      rootFiles.push(file)
                                    }
                                  })
                                  
                                  const folderEntries = Object.entries(filesByFolder)
                                  
                                  return (
                                    <>
                                      {/* Folders */}
                                      {folderEntries.map(([folderId, files]) => {
                                        const folderName = files[0]?.folder_name || "Folder"
                                        const folderFileIds = files.map(f => f.file_id)
                                        const selectedInFolder = folderFileIds.filter(id => selectedFiles.includes(id)).length
                                        const allFolderSelected = selectedInFolder === files.length
                                        const someFolderSelected = selectedInFolder > 0 && selectedInFolder < files.length
                                        
                                        return (
                                          <Collapsible key={folderId} open={expandedFolders.includes(folderId)} onOpenChange={() => toggleFolder(folderId)}>
                                            <div className="flex items-center gap-2 px-3 py-2 border-b border-border/50 bg-secondary/50 hover:bg-secondary/80 transition-colors">
                                              <Checkbox
                                                checked={allFolderSelected}
                                                ref={(el) => { if (el) (el as any).indeterminate = someFolderSelected }}
                                                onCheckedChange={(checked) => {
                                                  if (checked) {
                                                    setSelectedFiles(prev => [...new Set([...prev, ...folderFileIds])])
                                                  } else {
                                                    setSelectedFiles(prev => prev.filter(id => !folderFileIds.includes(id)))
                                                  }
                                                }}
                                              />
                                              <CollapsibleTrigger asChild>
                                                <button className="flex items-center gap-2 flex-1 text-left">
                                                  <ChevronRight className={cn("h-4 w-4 text-muted-foreground transition-transform", expandedFolders.includes(folderId) && "rotate-90")} />
                                                  <Folder className="h-4 w-4 text-yellow-500" />
                                                  <span className="text-sm font-medium flex-1 truncate">{folderName}</span>
                                                  <span className="text-xs text-muted-foreground">{selectedInFolder}/{files.length}</span>
                                                </button>
                                              </CollapsibleTrigger>
                                            </div>
                                            <CollapsibleContent>
                                              <div className="p-2 space-y-1 ml-4">
                                                {files.map((file) => {
                                                  const isSelected = selectedFiles.includes(file.file_id)
                                                  return (
                                                    <div
                                                      key={file.file_id}
                                                      onClick={() => toggleFileSelection(file.file_id)}
                                                      className={cn(
                                                        "flex items-center gap-2 rounded-lg border p-2 transition-colors cursor-pointer",
                                                        isSelected ? "border-green-500 bg-green-500/5" : "border-border bg-background hover:bg-secondary/50"
                                                      )}
                                                    >
                                                      <Checkbox checked={isSelected} onCheckedChange={() => toggleFileSelection(file.file_id)} onClick={(e) => e.stopPropagation()} />
                                                      <FileSpreadsheet className="h-4 w-4 text-primary flex-shrink-0" />
                                                      <div className="flex-1 min-w-0">
                                                        <p className="text-sm text-foreground break-words line-clamp-2 leading-tight">{file.filename}</p>
                                                        <span className="text-xs text-muted-foreground">{formatFileSize(file.size)}</span>
                                                      </div>
                                                    </div>
                                                  )
                                                })}
                                              </div>
                                            </CollapsibleContent>
                                          </Collapsible>
                                        )
                                      })}
                                      
                                      {/* Root files (no folder) */}
                                      {rootFiles.length > 0 && (
                                        <Collapsible open={expandedFolders.includes("files")} onOpenChange={() => toggleFolder("files")}>
                                          <div className="flex items-center gap-2 px-3 py-2 border-b border-border/50 bg-secondary/50 hover:bg-secondary/80 transition-colors">
                                            <Checkbox
                                              checked={rootFiles.every(f => selectedFiles.includes(f.file_id))}
                                              ref={(el) => { if (el) (el as any).indeterminate = rootFiles.some(f => selectedFiles.includes(f.file_id)) && !rootFiles.every(f => selectedFiles.includes(f.file_id)) }}
                                              onCheckedChange={(checked) => {
                                                const rootFileIds = rootFiles.map(f => f.file_id)
                                                if (checked) {
                                                  setSelectedFiles(prev => [...new Set([...prev, ...rootFileIds])])
                                                } else {
                                                  setSelectedFiles(prev => prev.filter(id => !rootFileIds.includes(id)))
                                                }
                                              }}
                                            />
                                            <CollapsibleTrigger asChild>
                                              <button className="flex items-center gap-2 flex-1 text-left">
                                                <ChevronRight className={cn("h-4 w-4 text-muted-foreground transition-transform", expandedFolders.includes("files") && "rotate-90")} />
                                                <FileSpreadsheet className="h-4 w-4 text-muted-foreground" />
                                                <span className="text-sm font-medium flex-1">Uploaded Files</span>
                                                <span className="text-xs text-muted-foreground">{rootFiles.filter(f => selectedFiles.includes(f.file_id)).length}/{rootFiles.length}</span>
                                              </button>
                                            </CollapsibleTrigger>
                                          </div>
                                          <CollapsibleContent>
                                            <div className="p-2 space-y-1">
                                              {rootFiles.map((file) => {
                                                const isSelected = selectedFiles.includes(file.file_id)
                                                return (
                                                  <div
                                                    key={file.file_id}
                                                    onClick={() => toggleFileSelection(file.file_id)}
                                                    className={cn(
                                                      "flex items-center gap-2 rounded-lg border p-2 transition-colors cursor-pointer",
                                                      isSelected ? "border-green-500 bg-green-500/5" : "border-border bg-background hover:bg-secondary/50"
                                                    )}
                                                  >
                                                    <Checkbox checked={isSelected} onCheckedChange={() => toggleFileSelection(file.file_id)} onClick={(e) => e.stopPropagation()} />
                                                    <FileSpreadsheet className="h-4 w-4 text-primary flex-shrink-0" />
                                                    <div className="flex-1 min-w-0">
                                                      <p className="text-sm text-foreground break-words line-clamp-2 leading-tight">{file.filename}</p>
                                                      <span className="text-xs text-muted-foreground">{formatFileSize(file.size)}</span>
                                                    </div>
                                                  </div>
                                                )
                                              })}
                                            </div>
                                          </CollapsibleContent>
                                        </Collapsible>
                                      )}
                                      
                                      {uploadedFiles.length === 0 && (
                                        <p className="text-xs text-muted-foreground px-3 py-4 text-center">No files uploaded yet. Go to Data Sources to upload.</p>
                                      )}
                                    </>
                                  )
                                })()}

                                <Collapsible open={expandedFolders.includes("databases")} onOpenChange={() => toggleFolder("databases")}>
                                  <CollapsibleTrigger asChild>
                                    <div className="flex items-center gap-2 px-3 py-2 border-b border-border/50 bg-secondary/50 hover:bg-secondary/80 transition-colors cursor-pointer">
                                      <Checkbox disabled />
                                      <ChevronRight className={cn("h-4 w-4 text-muted-foreground transition-transform", expandedFolders.includes("databases") && "rotate-90")} />
                                      <Server className="h-4 w-4 text-muted-foreground" />
                                      <span className="text-sm font-medium flex-1">Connected Databases</span>
                                      <span className="text-xs text-muted-foreground">{connections.length} databases</span>
                                    </div>
                                  </CollapsibleTrigger>
                                  <CollapsibleContent>
                                    <div className="p-2 space-y-2">
                                      {connections.length === 0 ? (
                                        <p className="text-xs text-muted-foreground px-2 py-4 text-center">No databases connected. Go to Data Sources to connect.</p>
                                      ) : (
                                        connections.map((conn) => (
                                          <div key={conn.id} className="space-y-1">
                                            <div
                                              onClick={() => toggleFileSelection(conn.id)}
                                              className={cn(
                                                "flex items-center gap-3 rounded-lg border p-2 transition-colors cursor-pointer",
                                                selectedFiles.includes(conn.id) ? "border-green-500 bg-green-500/5" : "border-border bg-background hover:bg-secondary/50"
                                              )}
                                            >
                                              <Checkbox checked={selectedFiles.includes(conn.id)} onCheckedChange={() => toggleFileSelection(conn.id)} onClick={(e) => e.stopPropagation()} />
                                              <div className="rounded-lg bg-secondary p-2">
                                                <Database className="h-4 w-4 text-blue-500" />
                                              </div>
                                              <div className="flex-1 min-w-0">
                                                <p className="font-medium text-sm text-foreground truncate">{conn.name}</p>
                                                <span className="text-xs text-muted-foreground">{conn.sub_type} {dbTables[conn.id] ? `• ${dbTables[conn.id].length} tables` : ""}</span>
                                              </div>
                                              {selectedFiles.includes(conn.id) && <Badge className="bg-green-500/10 text-green-600 border-green-500/30 text-xs">Selected</Badge>}
                                            </div>
                                            {selectedFiles.includes(conn.id) && dbTables[conn.id] && (
                                              <div className="ml-8 pl-2 border-l-2 border-blue-500/30 space-y-1">
                                                {dbTables[conn.id].map((table: string) => {
                                                  const tableId = `${conn.id}:${table}`
                                                  return (
                                                    <div 
                                                      key={table} 
                                                      onClick={(e) => { e.stopPropagation(); toggleFileSelection(tableId) }}
                                                      className={cn(
                                                        "flex items-center gap-2 py-1 px-2 text-xs rounded cursor-pointer",
                                                        selectedFiles.includes(tableId) ? "bg-green-500/10 text-green-600" : "text-muted-foreground hover:bg-secondary/50"
                                                      )}
                                                    >
                                                      <Checkbox 
                                                        checked={selectedFiles.includes(tableId)} 
                                                        onCheckedChange={() => toggleFileSelection(tableId)} 
                                                        onClick={(e) => e.stopPropagation()} 
                                                        className="h-3 w-3"
                                                      />
                                                      <Table2 className="h-3 w-3" />
                                                      <span>{table}</span>
                                                    </div>
                                                  )
                                                })}
                                              </div>
                                            )}
                                          </div>
                                        ))
                                      )}
                                    </div>
                                  </CollapsibleContent>
                                </Collapsible>
                              </div>
                            </div>
                            
                            <Collapsible>
                              <CollapsibleTrigger asChild>
                                <Button variant="ghost" className="w-full justify-between px-3 py-2 h-auto">
                                  <div className="flex items-center gap-2">
                                    <Settings className="h-4 w-4 text-muted-foreground" />
                                    <span className="text-sm font-medium">Fine-tune Settings</span>
                                  </div>
                                  <ChevronRight className="h-4 w-4 text-muted-foreground transition-transform group-data-[state=open]:rotate-90" />
                                </Button>
                              </CollapsibleTrigger>
                              <CollapsibleContent className="px-3 pb-3 space-y-3">
                                <div className="grid grid-cols-2 gap-3">
                                  <div className="space-y-1">
                                    <Label className="text-xs">Epochs</Label>
                                    <Input type="number" min={1} max={50} value={epochCount} onChange={(e) => setEpochCount(Number(e.target.value))} className="bg-secondary h-8 text-sm" />
                                  </div>
                                  <div className="space-y-1">
                                    <Label className="text-xs">Batch Size</Label>
                                    <Input type="number" min={8} max={256} step={8} value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} className="bg-secondary h-8 text-sm" />
                                  </div>
                                </div>
                                <div className="grid grid-cols-2 gap-3">
                                  <div className="space-y-1">
                                    <Label className="text-xs">Learning Rate</Label>
                                    <Input type="number" min={0.0001} max={0.1} step={0.0001} defaultValue={0.001} className="bg-secondary h-8 text-sm" />
                                  </div>
                                  <div className="space-y-1">
                                    <Label className="text-xs">Warmup Steps</Label>
                                    <Input type="number" min={0} max={1000} step={10} defaultValue={100} className="bg-secondary h-8 text-sm" />
                                  </div>
                                </div>
                              </CollapsibleContent>
                            </Collapsible>
                            <DialogFooter><Button onClick={handleFineTune} disabled={!newProjectName.trim() || selectedFiles.length === 0 || isTraining}>
                              <Sparkles className="mr-2 h-4 w-4" />
                              Start Fine-tuning
                            </Button></DialogFooter>
                          </>
                        ) : (
                          <div className="py-6 space-y-6">
                            <div className="text-center">
                              {trainingComplete ? (
                                <CheckCircle2 className="h-12 w-12 text-green-500 mx-auto mb-4" />
                              ) : (
                                <Loader2 className="h-12 w-12 text-primary mx-auto mb-4 animate-spin" />
                              )}
                              <h3 className="text-lg font-semibold">{trainingComplete ? "Training Complete!" : "Fine-tuning..."}</h3>
                              <p className="text-sm text-muted-foreground mt-1">{trainingStatus}</p>
                            </div>
                              {trainingComplete && mergedCsvPath && (
                                <a 
                                  href={"http://localhost:8080/uploads/" + mergedCsvPath.split("/").pop()}
                                  download
                                  className="mt-3 inline-flex items-center gap-2 px-3 py-1.5 bg-blue-500/10 text-blue-600 rounded-lg text-sm hover:bg-blue-500/20 transition-colors"
                                >
                                  <FileSpreadsheet className="h-4 w-4" />
                                  Download Merged CSV
                                </a>
                              )}
                            
                            <div className="space-y-4">
                              <div>
                                <div className="flex justify-between text-sm mb-2">
                                  <span>Progress</span>
                                  <span>{trainingProgress.toFixed(0)}%</span>
                                </div>
                                <Progress value={trainingProgress} className="h-2" />
                              </div>
                              
                              <div className="grid grid-cols-4 gap-2 text-center">
                                <div className="p-2 bg-secondary/50 rounded-lg">
                                  <div className="text-lg font-bold">{trainingEpoch}/{displayEpochs}</div>
                                  <div className="text-xs text-muted-foreground">Epoch</div>
                                </div>
                                <div className="p-2 bg-secondary/50 rounded-lg">
                                  <div className="text-lg font-bold">{trainingLoss.toFixed(3)}</div>
                                  <div className="text-xs text-muted-foreground">Loss</div>
                                </div>
                                <div className="p-2 bg-secondary/50 rounded-lg">
                                  <div className="text-lg font-bold text-green-600">{(trainingAccuracy * 100).toFixed(1)}%</div>
                                  <div className="text-xs text-muted-foreground">Accuracy</div>
                                </div>
                                <div className="p-2 bg-secondary/50 rounded-lg">
                                  <div className="text-lg font-bold text-blue-500">{trainingETA || "..."}</div>
                                  <div className="text-xs text-muted-foreground">ETA</div>
                                </div>
                              </div>

                              {lossHistory.length > 1 && (
                                <div className="mt-4 p-3 bg-secondary/30 rounded-lg">
                                  <p className="text-xs text-muted-foreground mb-2">Loss History</p>
                                  <ResponsiveContainer width="100%" height={120}>
                                    <LineChart data={lossHistory}>
                                      <XAxis dataKey="epoch" tick={{fontSize: 10}} />
                                      <YAxis tick={{fontSize: 10}} width={35} />
                                      <Tooltip contentStyle={{fontSize: 12, background: "hsl(var(--secondary))", border: "1px solid hsl(var(--border))"}} />
                                      <Line type="linear" dataKey="loss" stroke="hsl(var(--destructive))" strokeWidth={2} dot={false} name="Loss" />
                                      <Line type="monotone" dataKey="accuracy" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} name="Acc %" />
                                    </LineChart>
                                  </ResponsiveContainer>
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                      </TabsContent>
                      
                      <TabsContent value="models" className="space-y-4 mt-4">
                        <div className="space-y-2">
                          <Label>Project Name</Label>
                          <Input placeholder="e.g. Q4 Sales Analysis" value={newProjectName} onChange={(e) => setNewProjectName(e.target.value)} className="bg-secondary" />
                        </div>
                        
                        <div className="space-y-2">
                          <Label>Select a trained model</Label>
                          <div className="border rounded-lg max-h-[200px] overflow-y-auto bg-secondary/30">
                            {fineTunedModels.length === 0 ? (
                              <p className="text-sm text-muted-foreground p-4 text-center">No models available. Fine-tune a model first.</p>
                            ) : (
                              <TooltipProvider>
                                <div className="p-2 space-y-2">
                                  {fineTunedModels.map((model) => (
                                    <UITooltip key={model.id}>
                                      <TooltipTrigger asChild>
                                        <div
                                          onClick={() => !editingModelId && setSelectedExistingModel(model.id)}
                                          className={cn(
                                            "group flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-colors relative",
                                            selectedExistingModel === model.id ? "bg-primary/10 border border-primary/30" : "hover:bg-secondary/50 border border-transparent"
                                          )}
                                        >
                                          <Checkbox checked={selectedExistingModel === model.id} onCheckedChange={() => setSelectedExistingModel(model.id)} />
                                          <Sparkles className="h-4 w-4 text-purple-500" />
                                          <div className="flex flex-col flex-1 min-w-0">
                                            {editingModelId === model.id ? (
                                              <Input
                                                value={editingModelName}
                                                onChange={(e) => setEditingModelName(e.target.value)}
                                                onBlur={() => handleRenameModel(model.id)}
                                                onKeyDown={(e) => { if (e.key === "Enter") handleRenameModel(model.id); if (e.key === "Escape") { setEditingModelId(null); setEditingModelName(""); } }}
                                                onClick={(e) => e.stopPropagation()}
                                                className="h-6 text-sm"
                                                autoFocus
                                              />
                                            ) : (
                                              <span className="text-sm font-medium truncate">{model.name}</span>
                                            )}
                                            <span className="text-xs text-muted-foreground">v{model.version} - {(model.accuracy * 100).toFixed(1)}% accuracy</span>
                                          </div>
                                          <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1">
                                            <button onClick={(e) => { e.stopPropagation(); setEditingModelId(model.id); setEditingModelName(model.name); }} className="p-1 hover:bg-secondary rounded"><Pencil className="h-3 w-3" /></button>
                                            <button onClick={(e) => { e.stopPropagation(); handleDeleteModel(model.id); }} className="p-1 hover:bg-destructive/20 rounded text-destructive"><Trash2 className="h-3 w-3" /></button>
                                          </div>
                                        </div>
                                      </TooltipTrigger>
                                      <TooltipContent side="right" className="p-3">
                                        <div className="space-y-1 text-xs">
                                          <p><span className="text-muted-foreground">Accuracy:</span> {(model.accuracy * 100).toFixed(1)}%</p>
                                          <p><span className="text-muted-foreground">Epochs:</span> {model.epochs || "N/A"}</p>
                                          <p><span className="text-muted-foreground">Batch Size:</span> {model.batch_size || "N/A"}</p>
                                          <p><span className="text-muted-foreground">Loss:</span> {model.loss?.toFixed(4) || "N/A"}</p>
                                          <p><span className="text-muted-foreground">Created:</span> {new Date(model.created_at).toLocaleDateString()}</p>
                                        </div>
                                      </TooltipContent>
                                    </UITooltip>
                                  ))}
                                </div>
                              </TooltipProvider>
                            )}
                          </div>
                        </div>
                        
                        <div className="space-y-2">
                          <Label>Default Model</Label>
                          <Select value={selectedModel} onValueChange={setSelectedModel}>
                            <SelectTrigger className="bg-secondary">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="gpt-4o">GPT-4o</SelectItem>
                              <SelectItem value="gpt-4o-mini">GPT-4o Mini</SelectItem>
                              <SelectItem value="claude-sonnet-4-5">Claude Sonnet 4.5</SelectItem>
                              <SelectItem value="claude-opus-4">Claude Opus 4</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        
                        <DialogFooter><Button onClick={handleCreateProject} disabled={!newProjectName.trim() || !selectedExistingModel}>
                          Create Project
                        </Button></DialogFooter>
                      </TabsContent>
                    </Tabs>
                  </DialogContent>
                </Dialog>
              </SidebarMenuItem>
            </SidebarMenu>
            {state === "expanded" ? (
              <div className="flex-1 overflow-y-auto max-h-[360px] min-h-0 scrollbar-thin scrollbar-thumb-border scrollbar-track-transparent hover:scrollbar-thumb-muted-foreground/50">
                <SidebarMenu>
                  {recentQueries.map((query) => (
                    <SidebarMenuItem key={query.id}>
                      <ContextMenu>
                        <ContextMenuTrigger asChild>
                          {editingChatId === query.id ? (
                            <div className="px-2 py-1 pl-7">
                              <Input
                                value={editingName}
                                onChange={(e) => setEditingName(e.target.value)}
                                onBlur={saveEdit}
                                onKeyDown={(e) => {
                                  if (e.key === "Enter") saveEdit()
                                  if (e.key === "Escape") setEditingChatId(null)
                                }}
                                className="h-7 text-sm"
                                autoFocus
                              />
                            </div>
                          ) : query.isTraining ? (
                            <SidebarMenuButton 
                              isActive={false} 
                              tooltip={query.name + (trainingStates[query.id] ? ` - ${trainingStates[query.id].epoch}/${trainingStates[query.id].epochs} (${trainingStates[query.id].accuracy.toFixed(1)}%)` : " (Training...)")} 
                              className="pl-7" 
                              onClick={() => handleQueryClick(query)}
                            >
                              <Loader2 className="h-4 w-4 animate-spin mr-1" />
                              <span className="text-xs text-emerald-500 mr-1">
                                Training
                              </span>
                              <span className="truncate">{query.name}</span>
                            </SidebarMenuButton>
                          ) : (
                            <SidebarMenuButton 
                              isActive={pathname === "/playground/" + query.id} 
                              tooltip={query.name} 
                              className="pl-7"
                              onClick={() => handleQueryClick(query)}
                            >
                              <span className="truncate">{query.name}</span>
                            </SidebarMenuButton>
                          )}
                        </ContextMenuTrigger>
                        <ContextMenuContent className="w-48">
                          <ContextMenuItem onClick={() => startEditing(query)}>
                            <Pencil className="h-4 w-4 mr-2" />
                            Rename
                          </ContextMenuItem>
                          <ContextMenuItem onClick={() => handleDuplicateQuery(query)}>
                            <Copy className="h-4 w-4 mr-2" />
                            Duplicate
                          </ContextMenuItem>
                          <ContextMenuSeparator />
                          <ContextMenuItem onClick={() => handleDeleteQuery(query.id)} className="text-destructive">
                            <Trash2 className="h-4 w-4 mr-2" />
                            Delete
                          </ContextMenuItem>
                        </ContextMenuContent>
                      </ContextMenu>
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </div>
            ) : (
              queries.length > 0 && (
                <SidebarMenu>
                  <SidebarMenuItem>
                    <SidebarMenuButton asChild tooltip={queries.length + " queries"}>
                      <Link href={queries[0] ? "/playground/" + queries[0].id : "/playground"}>
                        <MessageSquare />
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                </SidebarMenu>
              )
            )}
          </SidebarGroup>

          <SidebarGroup>
            <SidebarGroupLabel>Configuration</SidebarGroupLabel>
            <SidebarMenu>
              {configItems.filter(item => !item.adminOnly || user?.role === "admin").map((item) => (
                <SidebarMenuItem key={item.href}>
                  <SidebarMenuButton asChild isActive={pathname === item.href} tooltip={item.label}>
                    <Link href={item.href}>
                      <item.icon />
                      <span>{item.label}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroup>
        </SidebarContent>

        <SidebarFooter>
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarTrigger className="w-full justify-start">
                <PanelLeft className="h-4 w-4" />
                <span>Toggle Sidebar</span>
              </SidebarTrigger>
            </SidebarMenuItem>
          </SidebarMenu>
          <NavUser user={{ name: user?.name || "Guest", email: user?.email || "", avatar: "" }} onLogout={logout} />
        </SidebarFooter>
      </SidebarPrimitive>

      {isProcessing && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/95 backdrop-blur-sm">
          <SchemaProcessingAnimation
            projectName={processingProjectName}
            dataSources={processingDataSources}
            isProcessing={isProcessing}
          />
        </div>
      )}
    </>
  )
}

export function Sidebar({ children }: { children?: React.ReactNode }) {
  return (
    <SidebarProvider>
      <SidebarInner />
      <main className="flex-1 flex flex-col overflow-auto">{children}</main>
    </SidebarProvider>
  )
}
