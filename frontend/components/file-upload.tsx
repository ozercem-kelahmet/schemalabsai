"use client"

import * as React from "react"
import { useCallback, useState } from "react"
import { Upload, File, X, CheckCircle2, AlertCircle, Loader2, FileSpreadsheet, FileText, FileJson } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { api } from "@/lib/api"

export type FileStatus = "idle" | "uploading" | "processing" | "done" | "error"

export interface UploadedFile {
  id: string
  name: string
  size: number
  type: string
  status: FileStatus
  progress: number
  error?: string
  fileId?: string
}

interface FileUploadProps {
  onFilesChange?: (files: UploadedFile[]) => void
  onUploadComplete?: (fileId: string, filename: string) => void
  maxFiles?: number
  maxSize?: number
  accept?: string[]
  className?: string
}

function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 Bytes"
  const k = 1024
  const sizes = ["Bytes", "KB", "MB", "GB"]
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
}

function getFileIcon(type: string) {
  if (type.includes("spreadsheet") || type.includes("csv") || type.includes("excel")) {
    return <FileSpreadsheet className="h-5 w-5 text-emerald-500" />
  }
  if (type.includes("json")) {
    return <FileJson className="h-5 w-5 text-amber-500" />
  }
  if (type.includes("pdf")) {
    return <FileText className="h-5 w-5 text-red-500" />
  }
  return <File className="h-5 w-5 text-muted-foreground" />
}

function getStatusIcon(status: FileStatus) {
  switch (status) {
    case "uploading":
    case "processing":
      return <Loader2 className="h-4 w-4 animate-spin text-primary" />
    case "done":
      return <CheckCircle2 className="h-4 w-4 text-emerald-500" />
    case "error":
      return <AlertCircle className="h-4 w-4 text-destructive" />
    default:
      return null
  }
}

function getStatusText(status: FileStatus, progress: number) {
  switch (status) {
    case "uploading":
      return `Uploading... ${progress}%`
    case "processing":
      return "Processing..."
    case "done":
      return "Complete"
    case "error":
      return "Failed"
    default:
      return "Ready"
  }
}

export function FileUpload({
  onFilesChange,
  onUploadComplete,
  maxFiles = 10,
  maxSize = 50 * 1024 * 1024,
  accept = [".csv", ".xlsx", ".xls", ".json", ".pdf", ".parquet"],
  className,
}: FileUploadProps) {
  const [files, setFiles] = useState<UploadedFile[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = React.useRef<HTMLInputElement>(null)

  const uploadFile = useCallback(async (file: File, uploadedFile: UploadedFile) => {
    try {
      setFiles((prev) => prev.map((f) => (f.id === uploadedFile.id ? { ...f, status: "uploading", progress: 50 } : f)))

      const result = await api.upload(file)

      setFiles((prev) => prev.map((f) => 
        f.id === uploadedFile.id 
          ? { ...f, status: "done", progress: 100, fileId: result.file_id } 
          : f
      ))

      if (onUploadComplete) {
        onUploadComplete(result.file_id, file.name)
      }
    } catch (error) {
      setFiles((prev) => prev.map((f) => 
        f.id === uploadedFile.id 
          ? { ...f, status: "error", error: "Upload failed" } 
          : f
      ))
    }
  }, [onUploadComplete])

  const addFiles = useCallback(
    (newFiles: FileList | null) => {
      if (!newFiles) return

      const fileArray = Array.from(newFiles).slice(0, maxFiles - files.length)
      const uploadedFiles: UploadedFile[] = fileArray.map((file) => ({
        id: crypto.randomUUID(),
        name: file.name,
        size: file.size,
        type: file.type,
        status: "uploading" as FileStatus,
        progress: 0,
      }))

      setFiles((prev) => {
        const updated = [...prev, ...uploadedFiles]
        onFilesChange?.(updated)
        return updated
      })

      fileArray.forEach((file, index) => {
        uploadFile(file, uploadedFiles[index])
      })
    },
    [files.length, maxFiles, onFilesChange, uploadFile],
  )

  const removeFile = useCallback(
    (id: string) => {
      setFiles((prev) => {
        const updated = prev.filter((f) => f.id !== id)
        onFilesChange?.(updated)
        return updated
      })
    },
    [onFilesChange],
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      addFiles(e.dataTransfer.files)
    },
    [addFiles],
  )

  return (
    <div className={cn("space-y-4", className)}>
      <div
        className={cn(
          "relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors",
          isDragging
            ? "border-primary bg-primary/5"
            : "border-border bg-card hover:border-primary/50 hover:bg-accent/50",
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={accept.join(",")}
          className="hidden"
          onChange={(e) => addFiles(e.target.files)}
        />
        <div className="flex flex-col items-center text-center">
          <div className="rounded-full bg-secondary p-4 mb-4">
            <Upload className="h-6 w-6 text-muted-foreground" />
          </div>
          <p className="text-sm font-medium text-foreground mb-1">Drag and drop your files here</p>
          <p className="text-xs text-muted-foreground mb-4">
            CSV, Excel, JSON, PDF, Parquet up to {formatFileSize(maxSize)}
          </p>
          <Button variant="outline" size="sm" onClick={() => inputRef.current?.click()}>
            Browse Files
          </Button>
        </div>
      </div>

      {files.length > 0 && (
        <div className="space-y-2">
          {files.map((file) => (
            <div
              key={file.id}
              className={cn(
                "flex items-center gap-3 rounded-lg border p-3 transition-colors",
                file.status === "error"
                  ? "border-destructive/50 bg-destructive/5"
                  : file.status === "done"
                    ? "border-emerald-500/50 bg-emerald-500/5"
                    : "border-border bg-card",
              )}
            >
              {getFileIcon(file.type)}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-foreground truncate">{file.name}</p>
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <span>{formatFileSize(file.size)}</span>
                  <span>•</span>
                  <span
                    className={cn(
                      file.status === "error" && "text-destructive",
                      file.status === "done" && "text-emerald-500",
                    )}
                  >
                    {getStatusText(file.status, file.progress)}
                  </span>
                </div>
                {(file.status === "uploading" || file.status === "processing") && (
                  <div className="mt-2 h-1 w-full rounded-full bg-secondary overflow-hidden">
                    <div
                      className="h-full bg-primary transition-all duration-300"
                      style={{ width: `${file.progress}%` }}
                    />
                  </div>
                )}
              </div>
              <div className="flex items-center gap-2">
                {getStatusIcon(file.status)}
                <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => removeFile(file.id)}>
                  <X className="h-4 w-4" />
                  <span className="sr-only">Remove file</span>
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
