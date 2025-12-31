"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Label } from "@/components/ui/label"
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
  DialogClose,
} from "@/components/ui/dialog"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Sidebar } from "@/components/sidebar"
import {
  FileSpreadsheet,
  Database,
  Link2,
  MoreHorizontal,
  Search,
  Upload,
  RefreshCw,
  ChevronRight,
  Folder,
  FolderPlus,
  Trash2,
  FileText,
  Pencil,
  Sparkles,
  GripVertical,
  Loader2,
  AlertCircle,
  X,
  Cloud,
  HardDrive,
  Zap,
  Box,
  Layers,
  Triangle,
  Hexagon,
  CircleDot,
  Blocks,
  Eye,
  EyeOff,
  CheckCircle2, Check,
  XCircle,
  FileJson,
  Table2,
} from "lucide-react"
import { api } from "@/lib/api"

interface DataSource {
  id: number
  name: string
  type: string
  subType?: string
  size: number
  lastSync: string
  uploadedAt: string
  status: string
  folderId?: string | null
  fileId?: string
  trained?: boolean
  modelPath?: string
  accuracy?: number
  version?: number
  connectionConfig?: Record<string, any>
}

interface FolderType {
  id: string
  name: string
  isOpen: boolean
}

interface UploadingFile {
  name: string
  size: number
  progress: number
}

// Database connections
const databaseConnections = [
  { name: "PostgreSQL", icon: Database, color: "text-blue-500", defaultPort: 5432 },
  { name: "MySQL", icon: Database, color: "text-orange-500", defaultPort: 3306 },
  { name: "Supabase", icon: Zap, color: "text-emerald-500", defaultPort: 5432 },
  { name: "MongoDB", icon: Layers, color: "text-green-500", defaultPort: 27017 },
  { name: "Databricks", icon: HardDrive, color: "text-red-500", defaultPort: 443 },
  { name: "Snowflake", icon: Hexagon, color: "text-cyan-500", defaultPort: 443 },
]

// Vector DB connections
const vectorDBConnections = [
  { name: "Pinecone", icon: Triangle, color: "text-purple-500" },
  { name: "Weaviate", icon: Box, color: "text-green-400" },
  { name: "Chroma", icon: CircleDot, color: "text-yellow-500" },
  { name: "LanceDB", icon: Blocks, color: "text-blue-400" },
]

// Cloud storage connections
const cloudConnections = [
  { name: "Google Drive", icon: Cloud, color: "text-yellow-500" },
  { name: "AWS S3", icon: Cloud, color: "text-orange-500" },
  { name: "Google Cloud Storage", icon: Cloud, color: "text-blue-400" },
]

// API connections
const apiConnections = [
  { name: "REST API", icon: Link2, color: "text-indigo-500", description: "Connect via REST endpoint" },
  { name: "GraphQL", icon: Link2, color: "text-pink-500", description: "Connect via GraphQL endpoint" },
]

export default function DataSourcesPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [connectedSources, setConnectedSources] = useState<DataSource[]>([])
  const [folders, setFolders] = useState<FolderType[]>([])
  const [newFolderName, setNewFolderName] = useState("")
  const [isCreateFolderOpen, setIsCreateFolderOpen] = useState(false)
  const [editingFolderId, setEditingFolderId] = useState<string | null>(null)
  const [editingFolderName, setEditingFolderName] = useState("")
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null)
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null)
  const [selectedConnection, setSelectedConnection] = useState<any>(null)
  const [connectionDetailOpen, setConnectionDetailOpen] = useState(false)
  const [connectionTables, setConnectionTables] = useState<string[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [uploadingFiles, setUploadingFiles] = useState<UploadingFile[]>([])
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null)
  const [selectedFiles, setSelectedFiles] = useState<Set<number>>(new Set())
  const [actionSuccess, setActionSuccess] = useState<string | null>(null)
  const dragSourceId = useRef<number | null>(null)

  // Database dialog state
  const [dbDialogOpen, setDbDialogOpen] = useState(false)
  const [selectedDbType, setSelectedDbType] = useState<string | null>(null)
  const [dbHost, setDbHost] = useState("")
  const [dbPort, setDbPort] = useState("")
  const [dbName, setDbName] = useState("")
  const [dbUser, setDbUser] = useState("")
  const [dbPassword, setDbPassword] = useState("")
  const [dbConnectionName, setDbConnectionName] = useState("")
  const [showDbPassword, setShowDbPassword] = useState(false)
  const [isTestingDb, setIsTestingDb] = useState(false)
  const [dbTestResult, setDbTestResult] = useState<"success" | "error" | null>(null)

  // Vector DB dialog state
  const [vectorDialogOpen, setVectorDialogOpen] = useState(false)
  const [selectedVectorType, setSelectedVectorType] = useState<string | null>(null)
  const [vectorApiKey, setVectorApiKey] = useState("")
  const [vectorEndpoint, setVectorEndpoint] = useState("")
  const [vectorConnectionName, setVectorConnectionName] = useState("")

  // Cloud dialog state
  const [cloudDialogOpen, setCloudDialogOpen] = useState(false)
  const [selectedCloudType, setSelectedCloudType] = useState<string | null>(null)
  const [cloudConnectionName, setCloudConnectionName] = useState("")
  const [cloudApiKey, setCloudApiKey] = useState("")
  const [cloudBucket, setCloudBucket] = useState("")

  // API dialog state
  const [apiDialogOpen, setApiDialogOpen] = useState(false)
  const [selectedApiType, setSelectedApiType] = useState<string | null>(null)
  const [apiEndpoint, setApiEndpoint] = useState("")
  const [apiConnectionName, setApiConnectionName] = useState("")
  const [apiAuthToken, setApiAuthToken] = useState("")

  useEffect(() => {
    loadFiles()
  }, [])

  const loadFiles = async () => {
    try {
      const [filesData, modelsData, foldersData, connectionsData] = await Promise.all([
        api.getUploadedFiles(),
        api.getFineTunedModels(),
        api.getFolders(),
        api.getConnections().catch(() => ({ connections: [] }))
      ])
      const files = filesData.files || []
      const models = modelsData.models || []
      
      const sources: DataSource[] = files.map((f: any, idx: number) => {
        const model = models.find((m: any) => m.source_file_id === f.file_id)
        return {
          id: idx + 1,
          name: f.filename,
          type: f.filename.endsWith(".csv") ? "csv" : f.filename.endsWith(".json") ? "json" : f.filename.endsWith(".parquet") ? "parquet" : "excel",
          size: f.size || 0,
          lastSync: model ? new Date(model.created_at).toLocaleString() : "Just now",
          uploadedAt: f.created_at ? new Date(f.created_at).toLocaleString() : new Date().toLocaleString(),
          status: "active",
          folderId: f.folder_id || null,
          fileId: f.file_id,
          trained: !!model,
          modelPath: model?.model_path,
          accuracy: model?.accuracy,
          version: model?.version || 1,
        }
      })
      
      const loadedFolders = (foldersData.folders || []).map((f: any) => ({ ...f, isOpen: false }))
      
      // Add connections to sources
      const connectionSources: DataSource[] = (connectionsData.connections || []).map((c: any, idx: number) => ({
        id: 10000 + idx,
        name: c.name,
        type: c.type,
        subType: c.sub_type,
        size: 0,
        lastSync: new Date(c.created_at).toLocaleString(),
        uploadedAt: new Date(c.created_at).toLocaleString(),
        status: "active",
        folderId: null,
        fileId: c.id,
        connectionConfig: {
          host: c.host,
          port: c.port,
          database: c.database,
          user: c.username || c.user
        }
      }))
      
      sources.push(...connectionSources)
      setFolders(loadedFolders)
      setConnectedSources(sources)
    } catch (error) {
      console.error("Failed to load files:", error)
    }
  }

  const formatFileSize = (bytes: number) => {
    if (!bytes || bytes === 0) return "Unknown size"
    if (bytes < 1024) return bytes + " B"
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB"
    return (bytes / (1024 * 1024)).toFixed(1) + " MB"
  }

  const handleUpload = async (file: File) => {
    const uploadingFile: UploadingFile = { name: file.name, size: file.size, progress: 0 }
    setUploadingFiles(prev => [...prev, uploadingFile])
    
    const progressInterval = setInterval(() => {
      setUploadingFiles(prev => prev.map(f => 
        f.name === file.name ? { ...f, progress: Math.min(f.progress + 10, 90) } : f
      ))
    }, 100)
    
    try {
      await api.upload(file)
      clearInterval(progressInterval)
      setUploadingFiles(prev => prev.map(f => f.name === file.name ? { ...f, progress: 100 } : f))
      setTimeout(() => setUploadingFiles(prev => prev.filter(f => f.name !== file.name)), 500)
      loadFiles()
    } catch (error) {
      console.error("Upload failed:", error)
      setUploadError(error instanceof Error ? error.message : "Upload failed")
      clearInterval(progressInterval)
      setUploadingFiles(prev => prev.filter(f => f.name !== file.name))
    }
  }

  const toggleFileSelect = (id: number, e: React.MouseEvent) => {
    e.stopPropagation()
    setSelectedFiles(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const selectAllFiles = () => {
    if (selectedFiles.size === connectedSources.length) {
      setSelectedFiles(new Set())
    } else {
      setSelectedFiles(new Set(connectedSources.map(s => s.id)))
    }
  }

  const moveSelectedToFolder = async (folderId: string | null) => {
    const count = selectedFiles.size
    for (const id of selectedFiles) {
      await moveToFolder(id, folderId)
    }
    setSelectedFiles(new Set())
    setActionSuccess(`Moved ${count} file${count > 1 ? "s" : ""} successfully`)
    setTimeout(() => setActionSuccess(null), 3000)
  }

  const deleteSelectedFiles = async () => {
    const count = selectedFiles.size
    for (const id of selectedFiles) {
      const source = connectedSources.find(s => s.id === id)
      if (source?.fileId) await deleteSource(id, source.fileId)
    }
    setSelectedFiles(new Set())
    setActionSuccess(`Deleted ${count} file${count > 1 ? "s" : ""} successfully`)
    setTimeout(() => setActionSuccess(null), 3000)
  }

  const handleMultipleUpload = async (files: File[]) => {
    setUploadError(null)
    setUploadSuccess(null)
    
    // Add all files to uploading state
    const uploadingList: UploadingFile[] = files.map(f => ({ name: f.name, size: f.size, progress: 0 }))
    setUploadingFiles(prev => [...prev, ...uploadingList])
    
    let successCount = 0
    let failCount = 0
    
    // Upload files in parallel
    await Promise.all(files.map(async (file) => {
      const progressInterval = setInterval(() => {
        setUploadingFiles(prev => prev.map(f => 
          f.name === file.name ? { ...f, progress: Math.min(f.progress + 10, 90) } : f
        ))
      }, 100)
      
      try {
        await api.upload(file)
        clearInterval(progressInterval)
        setUploadingFiles(prev => prev.map(f => f.name === file.name ? { ...f, progress: 100 } : f))
        successCount++
      } catch (error) {
        console.error("Upload failed:", file.name, error)
        clearInterval(progressInterval)
        failCount++
      }
    }))
    
    // Clear uploading files after a short delay
    setTimeout(() => {
      setUploadingFiles([])
      loadFiles()
    }, 500)
    
    // Show notification
    if (successCount > 0 && failCount === 0) {
      setUploadSuccess(`Successfully uploaded ${successCount} file${successCount > 1 ? "s" : ""}`)
      setTimeout(() => setUploadSuccess(null), 4000)
    } else if (failCount > 0) {
      setUploadError(`${failCount} file${failCount > 1 ? "s" : ""} failed to upload`)
    }
  }

  const createFolder = async () => {
    if (newFolderName.trim()) {
      try {
        const result = await api.createFolder(newFolderName.trim())
        setFolders([...folders, { id: result.id, name: result.name, isOpen: false }])
        setNewFolderName("")
        setIsCreateFolderOpen(false)
      } catch (error) {
        console.error("Failed to create folder:", error)
      }
    }
  }

  const toggleFolder = (folderId: string) => {
    setFolders(folders.map((f) => (f.id === folderId ? { ...f, isOpen: !f.isOpen } : f)))
  }

  const moveToFolder = async (sourceId: number, folderId: string | null) => {
    const source = connectedSources.find(s => s.id === sourceId)
    if (source?.fileId) {
      try {
        await api.moveFileToFolder(source.fileId, folderId)
        setConnectedSources(prev => prev.map((s) => (s.id === sourceId ? { ...s, folderId } : s)))
      } catch (error) {
        console.error("Failed to move file:", error)
      }
    }
  }

  const deleteSource = async (sourceId: number, fileId?: string) => {
    if (fileId) {
      try {
        await api.deleteFile(fileId)
      } catch (error) {
        console.error("Failed to delete file:", error)
      }
    }
    setConnectedSources(connectedSources.filter((s) => s.id !== sourceId))
  }

  const startEditingFolder = (folder: FolderType) => {
    setEditingFolderId(folder.id)
    setEditingFolderName(folder.name)
  }

  const saveEditFolder = async () => {
    if (editingFolderId && editingFolderName.trim()) {
      try {
        await api.updateFolder(editingFolderId, editingFolderName.trim())
        setFolders(folders.map(f => f.id === editingFolderId ? {...f, name: editingFolderName.trim()} : f))
      } catch (error) {
        console.error("Failed to update folder:", error)
      }
    }
    setEditingFolderId(null)
    setEditingFolderName("")
  }

  const handleDeleteFolder = async (folderId: string) => {
    try {
      await api.deleteFolder(folderId)
      setConnectedSources(connectedSources.map((s) => (s.folderId === folderId ? { ...s, folderId: null } : s)))
      setFolders(folders.filter((f) => f.id !== folderId))
    } catch (error) {
      console.error("Failed to delete folder:", error)
    }
  }

  const selectSource = async (source: DataSource) => {
    if (source.fileId) {
      setSelectedFileId(source.fileId)
      setSelectedFileName(source.name)
    }
    
    // If it's a database connection, load tables and show detail
    if (source.type === "database" || source.type === "vectordb") {
      setSelectedConnection(source)
      setConnectionDetailOpen(true)
      try {
        const result = await api.listTables(source.fileId!)
        setConnectionTables(result.tables || [])
      } catch (error) {
        console.error("Failed to load tables:", error)
        setConnectionTables([])
      }
    }
  }

  const testDatabaseConnection = async () => {
    setIsTestingDb(true)
    setDbTestResult(null)
    try {
      // Simulate test - replace with actual API call
      await new Promise(r => setTimeout(r, 1500))
      setDbTestResult("success")
    } catch {
      setDbTestResult("error")
    }
    setIsTestingDb(false)
  }

  const connectDatabase = async () => {
    try {
      const result = await api.createConnection({
        name: dbConnectionName || selectedDbType + " Connection",
        type: "database",
        sub_type: selectedDbType?.toLowerCase() || "",
        host: dbHost,
        port: parseInt(dbPort) || 5432,
        database: dbName,
        username: dbUser,
        password: dbPassword
      })
      if (result.id) {
        const newSource: DataSource = {
          id: Date.now(),
          name: result.name,
          type: "database",
          subType: result.sub_type,
          size: 0,
          lastSync: "Just now",
          uploadedAt: new Date().toLocaleString(),
          status: "active",
          folderId: null,
          fileId: result.id
        }
        setConnectedSources(prev => [...prev, newSource])
      }
      resetDbDialog()
    } catch (error) {
      console.error("Failed to create connection:", error)
    }
  }

  const resetDbDialog = () => {
    setDbDialogOpen(false)
    setSelectedDbType(null)
    setDbHost("")
    setDbPort("")
    setDbName("")
    setDbUser("")
    setDbPassword("")
    setDbConnectionName("")
    setDbTestResult(null)
  }

  const connectVectorDB = async () => {
    try {
      const result = await api.createConnection({
        name: vectorConnectionName || selectedVectorType + " Connection",
        type: "vectordb",
        sub_type: selectedVectorType?.toLowerCase() || "",
        api_key: vectorApiKey,
        endpoint: vectorEndpoint
      })
      if (result.id) {
        const newSource: DataSource = {
          id: Date.now(),
          name: result.name,
          type: "vectordb",
          subType: result.sub_type,
          size: 0,
          lastSync: "Just now",
          uploadedAt: new Date().toLocaleString(),
          status: "active",
          folderId: null,
          fileId: result.id
        }
        setConnectedSources(prev => [...prev, newSource])
      }
    } catch (error) {
      console.error("Failed to create connection:", error)
    }
    setVectorDialogOpen(false)
    setSelectedVectorType(null)
    setVectorApiKey("")
    setVectorEndpoint("")
    setVectorConnectionName("")
  }

  const connectCloud = async () => {
    try {
      const result = await api.createConnection({
        name: cloudConnectionName || selectedCloudType + " Connection",
        type: "cloud",
        sub_type: selectedCloudType?.toLowerCase().replace(" ", "_") || "",
        api_key: cloudApiKey,
        bucket: cloudBucket
      })
      if (result.id) {
        const newSource: DataSource = {
          id: Date.now(),
          name: result.name,
          type: "cloud",
          subType: result.sub_type,
          size: 0,
          lastSync: "Just now",
          uploadedAt: new Date().toLocaleString(),
          status: "active",
          folderId: null,
          fileId: result.id
        }
        setConnectedSources(prev => [...prev, newSource])
      }
    } catch (error) {
      console.error("Failed to create connection:", error)
    }
    setCloudDialogOpen(false)
    setSelectedCloudType(null)
    setCloudConnectionName("")
    setCloudApiKey("")
    setCloudBucket("")
  }

  const connectApi = async () => {
    try {
      const result = await api.createConnection({
        name: apiConnectionName || selectedApiType + " Connection",
        type: "api",
        sub_type: selectedApiType?.toLowerCase().replace(" ", "_") || "",
        endpoint: apiEndpoint,
        api_key: apiAuthToken
      })
      if (result.id) {
        const newSource: DataSource = {
          id: Date.now(),
          name: result.name,
          type: "api",
          subType: result.sub_type,
          size: 0,
          lastSync: "Just now",
          uploadedAt: new Date().toLocaleString(),
          status: "active",
          folderId: null,
          fileId: result.id
        }
        setConnectedSources(prev => [...prev, newSource])
      }
    } catch (error) {
      console.error("Failed to create connection:", error)
    }
    setApiDialogOpen(false)
    setSelectedApiType(null)
    setApiEndpoint("")
    setApiConnectionName("")
    setApiAuthToken("")
  }

  const filteredSources = (sources: DataSource[]) =>
    sources.filter((s) => s.name.toLowerCase().includes(searchQuery.toLowerCase()))

  const rootSources = connectedSources.filter((s) => s.folderId === null)

  const getSourceIcon = (source: DataSource) => {
    if (source.type === "database") return <Database className="h-5 w-5 text-blue-500" />
    if (source.type === "vectordb") return <Triangle className="h-5 w-5 text-purple-500" />
    if (source.type === "cloud") return <Cloud className="h-5 w-5 text-yellow-500" />
    if (source.type === "api") return <Link2 className="h-5 w-5 text-indigo-500" />
    if (source.type === "json") return <FileJson className="h-5 w-5 text-green-500" />
    if (source.type === "parquet") return <Table2 className="h-5 w-5 text-orange-500" />
    return <FileSpreadsheet className="h-5 w-5 text-primary" />
  }

  const renderSourceItem = (source: DataSource) => (
    <div
      key={source.id}
      draggable="true"
      onDragStart={(e) => {
        // Eğer bu dosya seçiliyse, tüm seçili dosyaları taşı
        if (selectedFiles.has(source.id) && selectedFiles.size > 1) {
          e.dataTransfer.setData("text/plain", Array.from(selectedFiles).join(","))
        } else {
          e.dataTransfer.setData("text/plain", source.id.toString())
        }
        dragSourceId.current = source.id
        e.dataTransfer.effectAllowed = "move"
        setTimeout(() => setIsDragging(true), 0)
      }}
      onDragEnd={() => {
        dragSourceId.current = null
        setIsDragging(false)
      }}
      className={`flex items-center gap-4 rounded-lg border p-4 transition-colors cursor-move ${
        selectedFiles.has(source.id)
          ? "border-emerald-500 bg-emerald-500/10" 
          : selectedFileId === source.fileId 
            ? "border-green-500 bg-green-500/5" 
            : "border-border bg-secondary/30 hover:bg-secondary/50"
      }`}
    >
      <div 
        onClick={(e) => toggleFileSelect(source.id, e)}
        className={`h-5 w-5 rounded border-2 flex items-center justify-center cursor-pointer transition-colors ${
          selectedFiles.has(source.id) 
            ? "border-emerald-500 bg-emerald-500" 
            : "border-muted-foreground/30 hover:border-emerald-500/50"
        }`}
      >
        {selectedFiles.has(source.id) && <Check className="h-3 w-3 text-white" />}
      </div>
      <GripVertical className="h-5 w-5 text-muted-foreground flex-shrink-0" />
      <div className="flex-1 flex items-center gap-4 cursor-pointer" onClick={() => selectSource(source)}>
        <div className="rounded-lg bg-secondary p-2">
          {getSourceIcon(source)}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <p className="font-medium text-foreground truncate">{source.name}</p>
            {source.trained && (
              <Badge className="bg-green-500/10 text-green-600 border-green-500/30">
                <Sparkles className="h-3 w-3 mr-1" />
                Trained {source.accuracy && source.accuracy + "%"}
              </Badge>
            )}
            <Badge variant="default" className="bg-green-500/10 text-green-600 border-green-500/30">
              ✓ Active
            </Badge>
            {source.type === "database" && (
              <Badge variant="outline" className="text-blue-500 border-blue-500/30">{source.subType}</Badge>
            )}
            {source.type === "vectordb" && (
              <Badge variant="outline" className="text-purple-500 border-purple-500/30">{source.subType}</Badge>
            )}
            {source.type === "cloud" && (
              <Badge variant="outline" className="text-yellow-600 border-yellow-500/30">{source.subType}</Badge>
            )}
            {source.type === "api" && (
              <Badge variant="outline" className="text-indigo-500 border-indigo-500/30">{source.subType}</Badge>
            )}
          </div>
          <p className="text-sm text-muted-foreground">
            {source.size > 0 ? formatFileSize(source.size) + " · " : ""}Uploaded {source.uploadedAt}
          </p>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="icon" onClick={(e) => { e.stopPropagation(); loadFiles() }}>
          <RefreshCw className="h-4 w-4 text-muted-foreground" />
        </Button>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" onClick={(e) => e.stopPropagation()}>
              <MoreHorizontal className="h-4 w-4 text-muted-foreground" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => moveToFolder(source.id, null)}>
              <FileText className="h-4 w-4 mr-2" />
              Move to root
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            {folders.map((folder) => (
              <DropdownMenuItem key={folder.id} onClick={() => moveToFolder(source.id, folder.id)}>
                <Folder className="h-4 w-4 mr-2" />
                Move to {folder.name}
              </DropdownMenuItem>
            ))}
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => deleteSource(source.id, source.fileId)} className="text-destructive">
              <Trash2 className="h-4 w-4 mr-2" />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  )

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar>
        <div className="p-6 pt-12 space-y-6 w-full">
          <Card className="bg-card border-border">
            <CardHeader className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-foreground">Upload Data</CardTitle>
                  <CardDescription>Schema will automatically parse and understand your tabular data</CardDescription>
                </div>
                <div className="flex gap-2">
                  {/* Connect API Button */}
                  <Dialog open={apiDialogOpen} onOpenChange={setApiDialogOpen}>
                    <DialogTrigger asChild>
                      <Button variant="outline" size="sm">
                        <Link2 className="h-4 w-4 mr-2" />
                        Connect API
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="sm:max-w-lg">
                      <DialogHeader>
                        <DialogTitle>Connect an API</DialogTitle>
                        <DialogDescription>Connect via REST or GraphQL endpoint</DialogDescription>
                      </DialogHeader>
                      <div className="space-y-4 py-4">
                        <div className="grid grid-cols-2 gap-3">
                          {apiConnections.map((conn) => (
                            <Button
                              key={conn.name}
                              variant={selectedApiType === conn.name ? "default" : "outline"}
                              className="h-auto flex-col gap-2 py-4"
                              onClick={() => setSelectedApiType(conn.name)}
                            >
                              <conn.icon className={selectedApiType === conn.name ? "h-6 w-6" : "h-6 w-6 " + conn.color} />
                              <span className="text-xs">{conn.name}</span>
                            </Button>
                          ))}
                        </div>
                        {selectedApiType && (
                          <>
                            <div className="space-y-2">
                              <Label>Connection Name</Label>
                              <Input placeholder="e.g., My REST API" value={apiConnectionName} onChange={(e) => setApiConnectionName(e.target.value)} />
                            </div>
                            <div className="space-y-2">
                              <Label>API Endpoint URL</Label>
                              <Input placeholder="https://api.example.com/data" value={apiEndpoint} onChange={(e) => setApiEndpoint(e.target.value)} />
                            </div>
                            <div className="space-y-2">
                              <Label>Authentication Token (Optional)</Label>
                              <Input placeholder="Bearer token or API key" type="password" value={apiAuthToken} onChange={(e) => setApiAuthToken(e.target.value)} />
                            </div>
                          </>
                        )}
                      </div>
                      <DialogFooter>
                        <Button variant="outline" onClick={() => setApiDialogOpen(false)}>Cancel</Button>
                        <Button onClick={connectApi} disabled={!selectedApiType || !apiEndpoint}>Connect</Button>
                      </DialogFooter>
                    </DialogContent>
                  </Dialog>

                  {/* Connect Database Button */}
                  <Dialog open={dbDialogOpen} onOpenChange={setDbDialogOpen}>
                    <DialogTrigger asChild>
                      <Button variant="outline" size="sm">
                        <Database className="h-4 w-4 mr-2" />
                        Connect Database
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="sm:max-w-lg">
                      <DialogHeader>
                        <DialogTitle>Connect a Database</DialogTitle>
                        <DialogDescription>Choose a database to connect</DialogDescription>
                      </DialogHeader>
                      <Tabs defaultValue="databases" className="mt-4">
                        <TabsList className="grid w-full grid-cols-3">
                          <TabsTrigger value="databases">Databases</TabsTrigger>
                          <TabsTrigger value="vectordb">Vector DBs</TabsTrigger>
                          <TabsTrigger value="cloud">Cloud Storage</TabsTrigger>
                        </TabsList>
                        
                        <TabsContent value="databases" className="space-y-4 mt-4">
                          <div className="grid grid-cols-3 gap-3">
                            {databaseConnections.map((conn) => (
                              <Button
                                key={conn.name}
                                variant={selectedDbType === conn.name ? "default" : "outline"}
                                className="h-auto flex-col gap-2 py-4"
                                onClick={() => {
                                  setSelectedDbType(conn.name)
                                  setDbPort(conn.defaultPort?.toString() || "")
                                }}
                              >
                                <conn.icon className={selectedDbType === conn.name ? "h-6 w-6" : "h-6 w-6 " + conn.color} />
                                <span className="text-xs">{conn.name}</span>
                              </Button>
                            ))}
                          </div>
                          {selectedDbType && (
                            <>
                              <div className="space-y-2">
                                <Label>Connection Name</Label>
                                <Input placeholder="e.g., Production DB" value={dbConnectionName} onChange={(e) => setDbConnectionName(e.target.value)} />
                              </div>
                              <div className="grid grid-cols-2 gap-3">
                                <div className="space-y-2">
                                  <Label>Host</Label>
                                  <Input placeholder="localhost" value={dbHost} onChange={(e) => setDbHost(e.target.value)} />
                                </div>
                                <div className="space-y-2">
                                  <Label>Port</Label>
                                  <Input placeholder="5432" value={dbPort} onChange={(e) => setDbPort(e.target.value)} />
                                </div>
                              </div>
                              <div className="space-y-2">
                                <Label>Database Name</Label>
                                <Input placeholder="mydb" value={dbName} onChange={(e) => setDbName(e.target.value)} />
                              </div>
                              <div className="grid grid-cols-2 gap-3">
                                <div className="space-y-2">
                                  <Label>Username</Label>
                                  <Input placeholder="user" value={dbUser} onChange={(e) => setDbUser(e.target.value)} />
                                </div>
                                <div className="space-y-2">
                                  <Label>Password</Label>
                                  <div className="relative">
                                    <Input type={showDbPassword ? "text" : "password"} placeholder="••••••••" value={dbPassword} onChange={(e) => setDbPassword(e.target.value)} />
                                    <button type="button" onClick={() => setShowDbPassword(!showDbPassword)} className="absolute right-3 top-1/2 -translate-y-1/2">
                                      {showDbPassword ? <EyeOff className="h-4 w-4 text-muted-foreground" /> : <Eye className="h-4 w-4 text-muted-foreground" />}
                                    </button>
                                  </div>
                                </div>
                              </div>
                              <div className="flex items-center gap-2">
                                <Button variant="outline" onClick={testDatabaseConnection} disabled={isTestingDb}>
                                  {isTestingDb ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <RefreshCw className="h-4 w-4 mr-2" />}
                                  Test Connection
                                </Button>
                                {dbTestResult === "success" && <span className="text-green-500 flex items-center"><CheckCircle2 className="h-4 w-4 mr-1" />Connected</span>}
                                {dbTestResult === "error" && <span className="text-red-500 flex items-center"><XCircle className="h-4 w-4 mr-1" />Failed</span>}
                              </div>
                            </>
                          )}
                        </TabsContent>
                        
                        <TabsContent value="vectordb" className="space-y-4 mt-4">
                          <div className="grid grid-cols-2 gap-3">
                            {vectorDBConnections.map((conn) => (
                              <Button
                                key={conn.name}
                                variant={selectedVectorType === conn.name ? "default" : "outline"}
                                className="h-auto flex-col gap-2 py-4"
                                onClick={() => setSelectedVectorType(conn.name)}
                              >
                                <conn.icon className={selectedVectorType === conn.name ? "h-6 w-6" : "h-6 w-6 " + conn.color} />
                                <span className="text-xs">{conn.name}</span>
                              </Button>
                            ))}
                          </div>
                          {selectedVectorType && (
                            <>
                              <div className="space-y-2">
                                <Label>Connection Name</Label>
                                <Input placeholder="e.g., My Pinecone Index" value={vectorConnectionName} onChange={(e) => setVectorConnectionName(e.target.value)} />
                              </div>
                              <div className="space-y-2">
                                <Label>API Key</Label>
                                <Input type="password" placeholder="Your API key" value={vectorApiKey} onChange={(e) => setVectorApiKey(e.target.value)} />
                              </div>
                              <div className="space-y-2">
                                <Label>Endpoint / Environment</Label>
                                <Input placeholder="https://xxx.pinecone.io or us-east-1" value={vectorEndpoint} onChange={(e) => setVectorEndpoint(e.target.value)} />
                              </div>
                            </>
                          )}
                        </TabsContent>
                        
                        <TabsContent value="cloud" className="space-y-4 mt-4">
                          <div className="grid grid-cols-3 gap-3">
                            {cloudConnections.map((conn) => (
                              <Button
                                key={conn.name}
                                variant={selectedCloudType === conn.name ? "default" : "outline"}
                                className="h-auto flex-col gap-2 py-4"
                                onClick={() => setSelectedCloudType(conn.name)}
                              >
                                <conn.icon className={selectedCloudType === conn.name ? "h-6 w-6" : "h-6 w-6 " + conn.color} />
                                <span className="text-xs">{conn.name}</span>
                              </Button>
                            ))}
                          </div>
                          {selectedCloudType && (
                            <>
                              <div className="space-y-2">
                                <Label>Connection Name</Label>
                                <Input placeholder="e.g., My S3 Bucket" value={cloudConnectionName} onChange={(e) => setCloudConnectionName(e.target.value)} />
                              </div>
                              {selectedCloudType === "Google Drive" ? (
                                <div className="p-4 bg-secondary/50 rounded-lg text-center">
                                  <Cloud className="h-10 w-10 text-yellow-500 mx-auto mb-2" />
                                  <p className="text-sm text-muted-foreground mb-3">Connect your Google Drive account</p>
                                  <Button variant="outline" onClick={() => window.location.href = "http://localhost:8080/api/google/auth"}>
                                    <Cloud className="h-4 w-4 mr-2" />
                                    Sign in with Google
                                  </Button>
                                </div>
                              ) : (
                                <>
                                  <div className="space-y-2">
                                    <Label>{selectedCloudType === "AWS S3" ? "Access Key ID" : "Service Account Key"}</Label>
                                    <Input type="password" placeholder="Your access key" value={cloudApiKey} onChange={(e) => setCloudApiKey(e.target.value)} />
                                  </div>
                                  <div className="space-y-2">
                                    <Label>Bucket Name</Label>
                                    <Input placeholder="my-bucket" value={cloudBucket} onChange={(e) => setCloudBucket(e.target.value)} />
                                  </div>
                                </>
                              )}
                            </>
                          )}
                        </TabsContent>
                      </Tabs>
                      <DialogFooter className="mt-4">
                        <Button variant="outline" onClick={resetDbDialog}>Cancel</Button>
                        <Button 
                          onClick={() => {
                            if (selectedDbType) connectDatabase()
                            else if (selectedVectorType) connectVectorDB()
                            else if (selectedCloudType) connectCloud()
                          }}
                          disabled={!selectedDbType && !selectedVectorType && !selectedCloudType}
                        >
                          Connect
                        </Button>
                      </DialogFooter>
                    </DialogContent>
                  </Dialog>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-6 pt-0">
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${isDragging ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"}`}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(e) => {
                  e.preventDefault()
                  setIsDragging(false)
                  const files = e.dataTransfer.files
                  if (files && files.length > 0) {
                    handleMultipleUpload(Array.from(files))
                  }
                }}
              >
                <Upload className="h-10 w-10 text-muted-foreground mx-auto mb-4" />
                <p className="text-foreground font-medium mb-1">Drag and drop your files here</p>
                <p className="text-sm text-muted-foreground mb-4">CSV, Excel, JSON, PDF, Parquet up to 50 MB</p>
                <Button variant="outline" onClick={() => document.getElementById("file-upload")?.click()}>
                  Browse Files
                </Button>
                <input
                  id="file-upload"
                  type="file"
                  className="hidden"
                  accept=".csv,.xlsx,.xls,.json,.pdf,.parquet"
                  multiple
                  onChange={(e) => {
                    const files = e.target.files
                    if (files && files.length > 0) {
                      handleMultipleUpload(Array.from(files))
                    }
                    e.target.value = ""
                  }}
                />
              </div>

              {uploadError && (
                <div className="mt-4 p-4 bg-destructive/10 border border-destructive/30 rounded-lg flex items-start gap-3">
                  <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-sm font-medium text-destructive">Upload Failed</p>
                    <p className="text-sm text-destructive/80 mt-1">{uploadError}</p>
                  </div>
                  <button onClick={() => setUploadError(null)} className="text-destructive/60 hover:text-destructive">
                    <X className="h-4 w-4" />
                  </button>
                </div>
              )}
              
              {uploadSuccess && (
                <div className="mt-4 p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-lg flex items-start gap-3">
                  <CheckCircle2 className="h-5 w-5 text-emerald-500 flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-sm font-medium text-emerald-600">{uploadSuccess}</p>
                  </div>
                  <button onClick={() => setUploadSuccess(null)} className="text-emerald-500/60 hover:text-emerald-500">
                    <X className="h-4 w-4" />
                  </button>
                </div>
              )}
              
              {uploadingFiles.length > 0 && (
                <div className="mt-4 space-y-3">
                  {uploadingFiles.map((file) => (
                    <div key={file.name} className="flex items-center gap-3 p-3 bg-secondary/30 rounded-lg">
                      <Loader2 className="h-4 w-4 animate-spin text-primary" />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-1">
                          <p className="text-sm font-medium truncate">{file.name}</p>
                          <span className="text-xs text-muted-foreground">{file.progress}%</span>
                        </div>
                        <Progress value={file.progress} className="h-1" />
                        <p className="text-xs text-muted-foreground mt-1">{formatFileSize(file.size)}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="bg-card border-border">
            <CardHeader className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-foreground">Connected Sources</CardTitle>
                  <CardDescription>Manage your data sources and connections</CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Dialog open={isCreateFolderOpen} onOpenChange={setIsCreateFolderOpen}>
                    <DialogTrigger asChild>
                      <Button variant="outline" size="sm">
                        <FolderPlus className="h-4 w-4 mr-2" />
                        New Folder
                      </Button>
                    </DialogTrigger>
                    <DialogContent>
                      <DialogHeader>
                        <DialogTitle>Create New Folder</DialogTitle>
                        <DialogDescription>Organize your data sources into folders</DialogDescription>
                      </DialogHeader>
                      <div className="space-y-4 py-4">
                        <div className="space-y-2">
                          <Label>Folder Name</Label>
                          <Input
                            placeholder="e.g., Marketing Data"
                            value={newFolderName}
                            onChange={(e) => setNewFolderName(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && createFolder()}
                          />
                        </div>
                      </div>
                      <DialogFooter>
                        <DialogClose asChild>
                          <Button variant="outline">Cancel</Button>
                        </DialogClose>
                        <Button onClick={createFolder}>Create Folder</Button>
                      </DialogFooter>
                    </DialogContent>
                  </Dialog>
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                    <Input
                      placeholder="Search..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="w-64 pl-9 bg-secondary"
                    />
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-6 pt-0">

              {actionSuccess && (
                <div className="mb-4 p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-lg flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                  <span className="text-sm text-emerald-600">{actionSuccess}</span>
                </div>
              )}
              <div className="space-y-3">
                {isDragging && (
                  <div
                    onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = "move" }}
                    onDrop={(e) => {
                      e.preventDefault()
                      const data = e.dataTransfer.getData("text/plain")
                      const ids = data.split(",").map(Number).filter(Boolean)
                      ids.forEach(id => moveToFolder(id, null))
                      setSelectedFiles(new Set())
                      setIsDragging(false)
                      if (ids.length > 0) {
                        setActionSuccess(`Moved ${ids.length} file${ids.length > 1 ? "s" : ""} to root`)
                        setTimeout(() => setActionSuccess(null), 3000)
                      }
                    }}
                    className="border border-dashed border-slate-300/20 rounded-lg p-4 text-center text-muted-foreground bg-transparent"
                  >
                    Drop here to move to ROOT
                  </div>
                )}
                
                {folders.map((folder) => (
                  <Collapsible key={folder.id} open={folder.isOpen} onOpenChange={() => toggleFolder(folder.id)}>
                    <div
                      onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = "move" }}
                      onDrop={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        const data = e.dataTransfer.getData("text/plain")
                        const ids = data.split(",").map(Number).filter(Boolean)
                        ids.forEach(id => moveToFolder(id, folder.id))
                        setSelectedFiles(new Set())
                        setIsDragging(false)
                        if (ids.length > 0) {
                          setActionSuccess(`Moved ${ids.length} file${ids.length > 1 ? "s" : ""} to ${folder.name}`)
                          setTimeout(() => setActionSuccess(null), 3000)
                        }
                      }}
                      className={`rounded-lg border-2 transition-all ${isDragging ? "border-dashed border-slate-300/20 bg-transparent" : "border-transparent"}`}
                    >
                      <div className="flex items-center gap-2 rounded-lg border border-border bg-muted/50 p-3">
                        <CollapsibleTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-6 w-6">
                            <ChevronRight className={`h-4 w-4 transition-transform ${folder.isOpen ? "rotate-90" : ""}`} />
                          </Button>
                        </CollapsibleTrigger>
                        <Folder className="h-5 w-5 text-muted-foreground" />
                        {editingFolderId === folder.id ? (
                          <Input
                            value={editingFolderName}
                            onChange={(e) => setEditingFolderName(e.target.value)}
                            onBlur={saveEditFolder}
                            onKeyDown={(e) => { if (e.key === "Enter") saveEditFolder(); if (e.key === "Escape") setEditingFolderId(null); }}
                            className="h-7 w-40 text-sm"
                            autoFocus
                          />
                        ) : (
                          <span className="font-medium text-foreground flex-1">{folder.name}</span>
                        )}
                        <Badge variant="outline" className="text-xs">
                          {connectedSources.filter((s) => s.folderId === folder.id).length} items
                        </Badge>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-6 w-6">
                              <MoreHorizontal className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem onClick={() => startEditingFolder(folder)}>
                              <Pencil className="h-4 w-4 mr-2" />
                              Rename
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => handleDeleteFolder(folder.id)} className="text-destructive">
                              <Trash2 className="h-4 w-4 mr-2" />
                              Delete Folder
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                      <CollapsibleContent>
                        <div className="ml-6 mt-2 space-y-2 border-l-2 border-border pl-4">
                          {filteredSources(connectedSources.filter((s) => s.folderId === folder.id)).map(renderSourceItem)}
                        </div>
                      </CollapsibleContent>
                    </div>
                  </Collapsible>
                ))}
                
                {filteredSources(rootSources).map(renderSourceItem)}
                
                {connectedSources.length === 0 && (
                  <div className="text-center py-8 text-muted-foreground">
                    <Database className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>No data sources connected yet</p>
                    <p className="text-sm">Upload a file or connect a database to get started</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Connection Detail Modal */}
          <Dialog open={connectionDetailOpen} onOpenChange={setConnectionDetailOpen}>
            <DialogContent className="sm:max-w-lg">
              <DialogHeader>
                <DialogTitle>Connection Details</DialogTitle>
                <DialogDescription>View and manage your database connection</DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="flex items-center gap-3 p-3 bg-secondary/30 rounded-lg">
                  <div className="rounded-lg bg-blue-500/10 p-3">
                    <Database className="h-6 w-6 text-blue-500" />
                  </div>
                  <div>
                    <p className="font-semibold">{selectedConnection?.name}</p>
                    <p className="text-sm text-muted-foreground">{selectedConnection?.subType} • {connectionTables.length} tables</p>
                  </div>
                  <Badge className="ml-auto bg-green-500/10 text-green-600 border-green-500/30">Connected</Badge>
                </div>

                <div className="space-y-2">
                  <Label>Connection Name</Label>
                  <Input value={selectedConnection?.name || ""} readOnly className="bg-secondary" />
                </div>
                
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <Label>Host</Label>
                    <Input value={selectedConnection?.connectionConfig?.host || ""} readOnly className="bg-secondary" />
                  </div>
                  <div className="space-y-2">
                    <Label>Port</Label>
                    <Input value={selectedConnection?.connectionConfig?.port || ""} readOnly className="bg-secondary" />
                  </div>
                </div>
                
                <div className="space-y-2">
                  <Label>Database Name</Label>
                  <Input value={selectedConnection?.connectionConfig?.database || ""} readOnly className="bg-secondary" />
                </div>
                
                <div className="space-y-2">
                  <Label>Username</Label>
                  <Input value={selectedConnection?.connectionConfig?.user || ""} readOnly className="bg-secondary" />
                </div>
                
                <div className="space-y-2">
                  <Label>Tables ({connectionTables.length})</Label>
                  <div className="max-h-32 overflow-y-auto border rounded-lg p-2 space-y-1 bg-secondary/30">
                    {connectionTables.length === 0 ? (
                      <p className="text-sm text-muted-foreground text-center py-4">Loading tables...</p>
                    ) : (
                      connectionTables.map((table) => (
                        <div key={table} className="flex items-center gap-2 p-1.5 hover:bg-secondary rounded text-sm">
                          <Table2 className="h-3 w-3 text-muted-foreground" />
                          <span>{table}</span>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setConnectionDetailOpen(false)}>Close</Button>
                <Button variant="destructive" onClick={async () => {
                  if (selectedConnection?.fileId) {
                    await api.deleteConnection(selectedConnection.fileId)
                    setConnectedSources(prev => prev.filter(s => s.fileId !== selectedConnection.fileId))
                    setConnectionDetailOpen(false)
                  }
                }}>
                  <Trash2 className="h-4 w-4 mr-2" />
                  Delete
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </Sidebar>
    </div>
  )
}
