'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Button } from '@/components/ui/button'
import { Upload, Link, FileText, Camera, User } from 'lucide-react'
import { useDropzone } from 'react-dropzone'

interface ProfileInputFormProps {
  onSubmit?: (data: { url?: string; file?: File; analysisType: string }) => Promise<void>
  onAnalyze?: (data: {
    profileUrl?: string
    files: File[]
    analysisType: 'url' | 'upload'
  }) => void
  isLoading?: boolean
}

export function ProfileInputForm({ onSubmit, onAnalyze, isLoading = false }: ProfileInputFormProps) {
  const [profileUrl, setProfileUrl] = useState('')
  const [analysisType, setAnalysisType] = useState<'url' | 'upload'>('url')
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (acceptedFiles: File[]) => {
      setSelectedFiles(acceptedFiles)
      setAnalysisType('upload')
    },
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'],
      'text/*': ['.txt', '.json', '.csv'],
      'application/json': ['.json']
    },
    multiple: true,
    maxFiles: 10
  })

    const handleSubmit = async () => {
    if (analysisType === 'url' && !profileUrl.trim()) {
      alert('Please enter a profile URL')
      return
    }
    
    if (analysisType === 'upload' && selectedFiles.length === 0) {
      alert('Please select files to analyze')
      return
    }

    if (onSubmit) {
      // New dashboard callback
      await onSubmit({
        url: analysisType === 'url' ? profileUrl : undefined,
        file: analysisType === 'upload' ? selectedFiles[0] : undefined,
        analysisType
      })
    } else if (onAnalyze) {
      // Legacy callback
      onAnalyze({
        profileUrl: analysisType === 'url' ? profileUrl : undefined,
        files: analysisType === 'upload' ? selectedFiles : [],
        analysisType
      })
    }
  }

  const clearFiles = () => {
    setSelectedFiles([])
    if (analysisType === 'upload') {
      setAnalysisType('url')
    }
  }

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <User className="h-6 w-6 text-blue-600" />
          Profile Analysis
        </CardTitle>
        <CardDescription>
          Analyze social media profiles for trustworthiness using AI-powered analysis
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Analysis Type Selector */}
        <div className="flex gap-4">
          <Button
            type="button"
            variant={analysisType === 'url' ? 'default' : 'outline'}
            onClick={() => setAnalysisType('url')}
            className="flex-1"
            disabled={isLoading}
          >
            <Link className="h-4 w-4 mr-2" />
            Profile URL
          </Button>
          <Button
            type="button"
            variant={analysisType === 'upload' ? 'default' : 'outline'}
            onClick={() => setAnalysisType('upload')}
            className="flex-1"
            disabled={isLoading}
          >
            <Upload className="h-4 w-4 mr-2" />
            Upload Files
          </Button>
        </div>

        {/* URL Input Form */}
        {analysisType === 'url' && (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="profileUrl">Profile URL</Label>
              <Input
                id="profileUrl"
                type="url"
                placeholder="https://twitter.com/username or https://instagram.com/username"
                value={profileUrl}
                onChange={(e) => setProfileUrl(e.target.value)}
                disabled={isLoading}
                className="w-full"
              />
              <p className="text-sm text-muted-foreground">
                Supported platforms: Twitter, Instagram, LinkedIn, Facebook
              </p>
            </div>
            
            <Button 
              type="submit" 
              className="w-full" 
              disabled={isLoading || !profileUrl.trim()}
              size="lg"
            >
              {isLoading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                  Analyzing Profile...
                </>
              ) : (
                <>
                  <User className="h-4 w-4 mr-2" />
                  Analyze Profile
                </>
              )}
            </Button>
          </form>
        )}

        {/* File Upload Form */}
        {analysisType === 'upload' && (
          <div className="space-y-4">
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                isDragActive
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-950'
                  : 'border-gray-300 hover:border-gray-400 dark:border-gray-600'
              }`}
            >
              <input {...getInputProps()} />
              <div className="flex flex-col items-center gap-4">
                <div className="flex gap-4">
                  <div className="p-3 rounded-full bg-blue-100 dark:bg-blue-900">
                    <FileText className="h-6 w-6 text-blue-600" />
                  </div>
                  <div className="p-3 rounded-full bg-green-100 dark:bg-green-900">
                    <Camera className="h-6 w-6 text-green-600" />
                  </div>
                </div>
                
                {isDragActive ? (
                  <p className="text-blue-600 font-medium">Drop the files here...</p>
                ) : (
                  <div className="space-y-2">
                    <p className="text-gray-600 dark:text-gray-300">
                      Drag & drop profile data here, or click to select
                    </p>
                    <p className="text-sm text-gray-500">
                      Support: Images (JPG, PNG), Text files, JSON data
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Selected Files Display */}
            {selectedFiles.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium text-gray-900 dark:text-gray-100">
                    Selected Files ({selectedFiles.length})
                  </h4>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={clearFiles}
                    disabled={isLoading}
                  >
                    Clear All
                  </Button>
                </div>
                
                <div className="grid grid-cols-1 gap-2 max-h-40 overflow-y-auto">
                  {selectedFiles.map((file, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-3 p-3 border rounded-lg bg-gray-50 dark:bg-gray-800"
                    >
                      <div className="p-2 rounded bg-white dark:bg-gray-700">
                        {file.type.startsWith('image/') ? (
                          <Camera className="h-4 w-4 text-green-600" />
                        ) : (
                          <FileText className="h-4 w-4 text-blue-600" />
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{file.name}</p>
                        <p className="text-xs text-gray-500">
                          {(file.size / 1024).toFixed(1)} KB
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
                
                <Button
                  onClick={handleSubmit}
                  className="w-full"
                  disabled={isLoading}
                  size="lg"
                >
                  {isLoading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                      Analyzing Files...
                    </>
                  ) : (
                    <>
                      <Upload className="h-4 w-4 mr-2" />
                      Analyze {selectedFiles.length} File{selectedFiles.length !== 1 ? 's' : ''}
                    </>
                  )}
                </Button>
              </div>
            )}
          </div>
        )}

        {/* Analysis Features Info */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t">
          <div className="text-center space-y-2">
            <div className="p-3 rounded-full bg-blue-100 dark:bg-blue-900 w-fit mx-auto">
              <FileText className="h-5 w-5 text-blue-600" />
            </div>
            <h4 className="font-medium text-sm">Text Analysis</h4>
            <p className="text-xs text-muted-foreground">
              Sentiment, language patterns, authenticity
            </p>
          </div>
          
          <div className="text-center space-y-2">
            <div className="p-3 rounded-full bg-green-100 dark:bg-green-900 w-fit mx-auto">
              <Camera className="h-5 w-5 text-green-600" />
            </div>
            <h4 className="font-medium text-sm">Image Analysis</h4>
            <p className="text-xs text-muted-foreground">
              Face detection, deepfake analysis
            </p>
          </div>
          
          <div className="text-center space-y-2">
            <div className="p-3 rounded-full bg-purple-100 dark:bg-purple-900 w-fit mx-auto">
              <User className="h-5 w-5 text-purple-600" />
            </div>
            <h4 className="font-medium text-sm">Profile Metrics</h4>
            <p className="text-xs text-muted-foreground">
              Account age, activity patterns
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}