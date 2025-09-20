'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { User, BarChart3 } from 'lucide-react'

interface ProfileData {
  username: string
  displayName: string
  bio: string
  profileText: string
  followerCount: number
  followingCount: number
  postCount: number
  accountAge?: number // Optional field
  platform: string
  verified: boolean
}

interface ProfileInputFormProps {
  onSubmit?: (data: ProfileData) => Promise<void>
  isLoading?: boolean
}

export function ProfileInputForm({ onSubmit, isLoading = false }: ProfileInputFormProps) {
  const [formData, setFormData] = useState<ProfileData>({
    username: '',
    displayName: '',
    bio: '',
    profileText: '',
    followerCount: 0,
    followingCount: 0,
    postCount: 0,
    accountAge: undefined,
    platform: '',
    verified: false
  })

  const [errors, setErrors] = useState<Record<string, string>>({})

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {}

    // Required fields validation
    if (!formData.username.trim()) newErrors.username = 'Username is required'
    if (!formData.displayName.trim()) newErrors.displayName = 'Display name is required'
    if (!formData.bio.trim()) newErrors.bio = 'Bio/Description is required'
    if (!formData.profileText.trim()) newErrors.profileText = 'Profile text content is required'
    if (!formData.platform) newErrors.platform = 'Platform is required'
    
    // Numeric validation
    if (formData.followerCount < 0) newErrors.followerCount = 'Follower count cannot be negative'
    if (formData.followingCount < 0) newErrors.followingCount = 'Following count cannot be negative'
    if (formData.postCount < 0) newErrors.postCount = 'Post count cannot be negative'
    if (formData.accountAge !== undefined && formData.accountAge < 0) {
      newErrors.accountAge = 'Account age cannot be negative'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!validateForm()) {
      return
    }

    if (onSubmit) {
      await onSubmit(formData)
    }
  }

  const updateField = (field: keyof ProfileData, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }))
    
    // Clear error for this field when user starts typing
    if (errors[field]) {
      setErrors(prev => ({
        ...prev,
        [field]: ''
      }))
    }
  }

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <User className="h-6 w-6 text-blue-600" />
          Profile Purity Analysis
        </CardTitle>
        <CardDescription>
          Analyze social media profiles using multi-faceted AI detection (NLP, Computer Vision, Profile Metrics)
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

        {/* Automatic Extraction Features */}
        <Card className="bg-green-50 dark:bg-green-950 border-green-200 dark:border-green-800">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Zap className="h-5 w-5 text-green-600" />
              Automatic Profile Extraction
            </CardTitle>
            <CardDescription>
              Simply provide a profile URL - all metrics will be extracted automatically!
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <span className="text-sm">Follower & following counts</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <span className="text-sm">Post counts & engagement</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <span className="text-sm">Profile bio & description</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <span className="text-sm">Account age & verification</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* URL Input Form */}
        {analysisType === 'url' && (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="profileUrl">Social Media Profile URL</Label>
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
                Supported platforms: Instagram, Twitter/X, Facebook - All data extracted automatically!
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
                  Step 1: Extracting Profile Data...
                </>
              ) : (
                <>
                  <Zap className="h-4 w-4 mr-2" />
                  Auto-Extract & Analyze Profile
                </>
              )}
            </Button>
            
            {isLoading && (
              <Card className="bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800">
                <CardContent className="pt-6">
                  <div className="space-y-3">
                    <div className="flex items-center gap-3">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600" />
                      <span className="text-sm">Extracting profile data from URL...</span>
                    </div>
                    <div className="flex items-center gap-3 text-muted-foreground">
                      <div className="h-4 w-4 rounded-full border-2 border-gray-300" />
                      <span className="text-sm">Running NLP analysis...</span>
                    </div>
                    <div className="flex items-center gap-3 text-muted-foreground">
                      <div className="h-4 w-4 rounded-full border-2 border-gray-300" />
                      <span className="text-sm">Processing computer vision...</span>
                    </div>
                    <div className="flex items-center gap-3 text-muted-foreground">
                      <div className="h-4 w-4 rounded-full border-2 border-gray-300" />
                      <span className="text-sm">Calculating trust score...</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
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
            <h4 className="font-medium text-sm">NLP Analysis</h4>
            <p className="text-xs text-muted-foreground">
              Sentiment, grammar, coherence detection
            </p>
          </div>
          
          <div className="text-center space-y-2">
            <div className="p-3 rounded-full bg-green-100 dark:bg-green-900 w-fit mx-auto">
              <Camera className="h-5 w-5 text-green-600" />
            </div>
            <h4 className="font-medium text-sm">Computer Vision</h4>
            <p className="text-xs text-muted-foreground">
              Stock photos, AI-generated face detection
            </p>
          </div>
          
          <div className="text-center space-y-2">
            <div className="p-3 rounded-full bg-purple-100 dark:bg-purple-900 w-fit mx-auto">
              <User className="h-5 w-5 text-purple-600" />
            </div>
            <h4 className="font-medium text-sm">Profile Metrics</h4>
            <p className="text-xs text-muted-foreground">
              Follower ratio, account age, activity patterns
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}