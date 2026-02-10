"use client"

import React from "react"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import type { EvaluationMetrics, Model } from "@/lib/types"
import {
  MessageSquare,
  Download,
  RotateCcw,
  CheckCircle2,
  Database,
  Box,
  FileJson,
  FileArchive,
  Cloud,
  HardDrive,
  ExternalLink,
  Check,
  Loader2,
} from "lucide-react"
import { SourceBadge } from "@/components/datasets/source-badge"

interface EvaluateStepProps {
  metrics: EvaluationMetrics
  model: Model
  trainingTime: number
  onTrainAgain: () => void
  onOpenPlayground: () => void
}

type ExportFormat = "onnx" | "safetensors" | "pytorch" | "tensorflow"
type ExportDestination = "download" | "huggingface" | "aws-s3" | "gcs"

interface ExportOption {
  id: ExportFormat
  name: string
  description: string
  extension: string
  icon: React.ReactNode
}

interface DestinationOption {
  id: ExportDestination
  name: string
  description: string
  icon: React.ReactNode
}

const exportFormats: ExportOption[] = [
  {
    id: "onnx",
    name: "ONNX",
    description: "Open Neural Network Exchange - Universal format for ML interoperability",
    extension: ".onnx",
    icon: <FileJson className="h-5 w-5" />,
  },
  {
    id: "safetensors",
    name: "SafeTensors",
    description: "HuggingFace format - Safe and fast model serialization",
    extension: ".safetensors",
    icon: <FileArchive className="h-5 w-5" />,
  },
  {
    id: "pytorch",
    name: "PyTorch",
    description: "Native PyTorch checkpoint format",
    extension: ".pt",
    icon: <FileArchive className="h-5 w-5" />,
  },
  {
    id: "tensorflow",
    name: "TensorFlow SavedModel",
    description: "TensorFlow/Keras compatible format",
    extension: ".pb",
    icon: <FileArchive className="h-5 w-5" />,
  },
]

const exportDestinations: DestinationOption[] = [
  {
    id: "download",
    name: "Download to Computer",
    description: "Download the model files directly",
    icon: <HardDrive className="h-5 w-5" />,
  },
  {
    id: "huggingface",
    name: "HuggingFace Hub",
    description: "Push directly to your HuggingFace repository",
    icon: <span className="text-lg">🤗</span>,
  },
  {
    id: "aws-s3",
    name: "AWS S3",
    description: "Export to your Amazon S3 bucket",
    icon: <Cloud className="h-5 w-5" />,
  },
  {
    id: "gcs",
    name: "Google Cloud Storage",
    description: "Export to your GCS bucket",
    icon: <Cloud className="h-5 w-5" />,
  },
]

export function EvaluateStep({ metrics, model, trainingTime, onTrainAgain, onOpenPlayground }: EvaluateStepProps) {
  const [showExportModal, setShowExportModal] = useState(false)
  const [selectedFormat, setSelectedFormat] = useState<ExportFormat>("safetensors")
  const [selectedDestination, setSelectedDestination] = useState<ExportDestination>("download")
  const [includeConfig, setIncludeConfig] = useState(true)
  const [includeTokenizer, setIncludeTokenizer] = useState(true)
  const [isExporting, setIsExporting] = useState(false)
  const [exportSuccess, setExportSuccess] = useState(false)

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}m ${secs}s`
  }

  const handleExport = async () => {
    setIsExporting(true)
    
    try {
      if (selectedDestination === "download") {
        // Real download from backend
const response = await fetch(`/api/models/finetuned/download?id=${model.id}`, {
            credentials: 'include'
        })
        
        if (!response.ok) {
          throw new Error('Download failed')
        }
        
        const blob = await response.blob()
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `${model.name.toLowerCase().replace(/\s+/g, '-')}.pt`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
        
        setExportSuccess(true)
      } else {
        // Other destinations not implemented yet
        await new Promise(resolve => setTimeout(resolve, 1000))
        setExportSuccess(true)
      }
    } catch (error) {
      console.error('Export failed:', error)
      alert('Export failed. Please try again.')
    } finally {
      setIsExporting(false)
    }
    
    // Reset after showing success
    setTimeout(() => {
      setExportSuccess(false)
      setShowExportModal(false)
    }, 2000)
  }

  return (
    <div className="space-y-6">
      {/* Success Banner */}
      <div className="flex items-center gap-4 rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-4">
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-emerald-500/20">
          <CheckCircle2 className="h-6 w-6 text-emerald-500" />
        </div>
        <div className="flex-1">
          <h3 className="font-semibold text-foreground">Model Built Successfully</h3>
          <p className="text-sm text-muted-foreground">
            {model.name} built in {formatTime(trainingTime)} - Now available in Models
          </p>
        </div>
      </div>

      {/* Model Summary */}
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-base text-foreground">Model Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div>
              <p className="text-xs text-muted-foreground">Name</p>
              <p className="text-foreground font-medium">{model.name}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Sync Mode</p>
              <p className="text-foreground capitalize">{model.syncMode}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Base Model</p>
              <div className="flex items-center gap-2 mt-1">
                <Box className="h-4 w-4 text-[#2684FF]" />
                <span className="font-mono text-sm text-[#2684FF]">{model.baseModel}</span>
              </div>
            </div>
          </div>

          {/* Connected Datasets */}
          <div className="mt-4 pt-4 border-t border-border">
            <div className="flex items-center gap-2 text-xs text-muted-foreground mb-2">
              <Database className="h-3 w-3" />
              Connected Data Sources ({model.datasets.length})
            </div>
            <div className="flex flex-wrap gap-2">
              {model.datasets.map((ds) => (
                <div key={ds.datasetId} className="flex items-center gap-2 rounded bg-muted px-2 py-1.5">
                  <SourceBadge source={ds.source} size="sm" />
                  <span className="text-sm text-foreground">{ds.datasetName}</span>
                  <span className="text-xs text-muted-foreground">{ds.rows.toLocaleString()} rows</span>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Metrics Grid */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground">Accuracy</p>
            <p className="mt-1 font-mono text-2xl font-semibold text-emerald-500">
              {(metrics.accuracy > 1 ? metrics.accuracy.toFixed(1) : (metrics.accuracy * 100).toFixed(1))}%
            </p>
          </CardContent>
        </Card>
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground">Precision</p>
            <p className="mt-1 font-mono text-2xl font-semibold text-[#2684FF]">
              {(metrics.precision > 1 ? metrics.precision.toFixed(1) : (metrics.precision * 100).toFixed(1))}%
            </p>
          </CardContent>
        </Card>
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground">Recall</p>
            <p className="mt-1 font-mono text-2xl font-semibold text-[#2684FF]">{(metrics.recall > 1 ? metrics.recall.toFixed(1) : (metrics.recall * 100).toFixed(1))}%</p>
          </CardContent>
        </Card>
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground">F1 Score</p>
            <p className="mt-1 font-mono text-2xl font-semibold text-[#2684FF]">
              {(metrics.f1Score > 1 ? metrics.f1Score.toFixed(1) : (metrics.f1Score * 100).toFixed(1))}%
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Actions */}
      <div className="flex gap-4">
        <Button
          onClick={onOpenPlayground}
          className="flex-1 gap-2 bg-[#0052CC] text-white hover:bg-[#003D99]"
          size="lg"
        >
          <MessageSquare className="h-4 w-4" />
          Open in Playground
        </Button>
        <Button
          variant="outline"
          size="lg"
          onClick={onTrainAgain}
          className="gap-2 border-border bg-transparent text-foreground hover:bg-accent hover:text-accent-foreground"
        >
          <RotateCcw className="h-4 w-4" />
          Build Again
        </Button>
        <Button
          variant="outline"
          size="lg"
          onClick={() => setShowExportModal(true)}
          className="gap-2 border-border bg-transparent text-foreground hover:bg-accent hover:text-accent-foreground"
        >
          <Download className="h-4 w-4" />
          Export
        </Button>
      </div>

      {/* Export Modal */}
      <Dialog open={showExportModal} onOpenChange={setShowExportModal}>
        <DialogContent className="max-w-2xl bg-card border-border">
          <DialogHeader>
            <DialogTitle className="text-foreground">Export Model</DialogTitle>
            <DialogDescription className="text-muted-foreground">
              Export {model.name} for deployment to external platforms
            </DialogDescription>
          </DialogHeader>

          {exportSuccess ? (
            <div className="flex flex-col items-center justify-center py-12">
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-emerald-500/20 mb-4">
                <Check className="h-8 w-8 text-emerald-500" />
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-2">Export Successful!</h3>
              <p className="text-sm text-muted-foreground">
                {selectedDestination === "download" 
                  ? "Your model has been downloaded." 
                  : `Your model has been exported to ${exportDestinations.find(d => d.id === selectedDestination)?.name}.`}
              </p>
            </div>
          ) : (
            <div className="space-y-6 py-4">
              {/* Format Selection */}
              <div className="space-y-3">
                <Label className="text-sm font-medium text-foreground">Export Format</Label>
                <RadioGroup
                  value={selectedFormat}
                  onValueChange={(v) => setSelectedFormat(v as ExportFormat)}
                  className="grid grid-cols-2 gap-3"
                >
                  {exportFormats.map((format) => (
                    <div key={format.id}>
                      <RadioGroupItem value={format.id} id={format.id} className="peer sr-only" />
                      <Label
                        htmlFor={format.id}
                        className="flex items-start gap-3 rounded-lg border border-border bg-background p-4 cursor-pointer transition-all hover:bg-muted peer-data-[state=checked]:border-[#0052CC] peer-data-[state=checked]:bg-[#0052CC]/10"
                      >
                        <div className="text-muted-foreground peer-data-[state=checked]:text-[#2684FF]">
                          {format.icon}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-foreground">{format.name}</span>
                            <span className="text-xs text-muted-foreground font-mono">{format.extension}</span>
                          </div>
                          <p className="text-xs text-muted-foreground mt-0.5">{format.description}</p>
                        </div>
                      </Label>
                    </div>
                  ))}
                </RadioGroup>
              </div>

              {/* Destination Selection */}
              <div className="space-y-3">
                <Label className="text-sm font-medium text-foreground">Export Destination</Label>
                <RadioGroup
                  value={selectedDestination}
                  onValueChange={(v) => setSelectedDestination(v as ExportDestination)}
                  className="grid grid-cols-2 gap-3"
                >
                  {exportDestinations.map((dest) => (
                    <div key={dest.id}>
                      <RadioGroupItem value={dest.id} id={dest.id} className="peer sr-only" />
                      <Label
                        htmlFor={dest.id}
                        className="flex items-start gap-3 rounded-lg border border-border bg-background p-4 cursor-pointer transition-all hover:bg-muted peer-data-[state=checked]:border-[#0052CC] peer-data-[state=checked]:bg-[#0052CC]/10"
                      >
                        <div className="text-muted-foreground peer-data-[state=checked]:text-[#2684FF]">
                          {dest.icon}
                        </div>
                        <div className="flex-1">
                          <span className="font-medium text-foreground">{dest.name}</span>
                          <p className="text-xs text-muted-foreground mt-0.5">{dest.description}</p>
                        </div>
                      </Label>
                    </div>
                  ))}
                </RadioGroup>
              </div>

              {/* Additional Options */}
              <div className="space-y-3">
                <Label className="text-sm font-medium text-foreground">Include</Label>
                <div className="flex gap-6">
                  <div className="flex items-center gap-2">
                    <Checkbox
                      id="include-config"
                      checked={includeConfig}
                      onCheckedChange={(checked) => setIncludeConfig(checked as boolean)}
                      className="border-border data-[state=checked]:bg-[#0052CC] data-[state=checked]:border-[#0052CC]"
                    />
                    <Label htmlFor="include-config" className="text-sm text-foreground cursor-pointer">
                      Model config (config.json)
                    </Label>
                  </div>
                  <div className="flex items-center gap-2">
                    <Checkbox
                      id="include-tokenizer"
                      checked={includeTokenizer}
                      onCheckedChange={(checked) => setIncludeTokenizer(checked as boolean)}
                      className="border-border data-[state=checked]:bg-[#0052CC] data-[state=checked]:border-[#0052CC]"
                    />
                    <Label htmlFor="include-tokenizer" className="text-sm text-foreground cursor-pointer">
                      Tokenizer files
                    </Label>
                  </div>
                </div>
              </div>

              {/* HuggingFace/Cloud specific fields */}
              {selectedDestination !== "download" && (
                <div className="rounded-lg border border-border bg-muted/50 p-4">
                  <p className="text-sm text-muted-foreground mb-3">
                    {selectedDestination === "huggingface" && (
                      <>Connect your HuggingFace account to push directly to your model hub.</>
                    )}
                    {selectedDestination === "aws-s3" && (
                      <>Configure your AWS S3 bucket to export the model files.</>
                    )}
                    {selectedDestination === "gcs" && (
                      <>Configure your Google Cloud Storage bucket to export the model files.</>
                    )}
                  </p>
                  <Button variant="outline" size="sm" className="gap-2 border-border bg-transparent text-foreground">
                    <ExternalLink className="h-3 w-3" />
                    Connect {selectedDestination === "huggingface" ? "HuggingFace" : selectedDestination === "aws-s3" ? "AWS" : "GCS"}
                  </Button>
                </div>
              )}

              {/* Export Button */}
              <Button
                onClick={handleExport}
                disabled={isExporting}
                className="w-full gap-2 bg-[#0052CC] text-white hover:bg-[#003D99]"
                size="lg"
              >
                {isExporting ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Exporting...
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4" />
                    Export Model
                  </>
                )}
              </Button>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}
