'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { User, BarChart3, Upload, Camera, X } from 'lucide-react'

interface ProfileData {
  username: string
  displayName: string
  platform: string
  bio: string
  profileText: string
  followerCount: number
  followingCount: number
  postCount: number
  accountAge?: number
  verified: boolean
  profilePicture?: File
}

interface AnalysisResult {
  trustScore: number
  nlpScore: number
  visionScore: number
  profileScore: number
  ensemble: {
    trust_score: number
    risk_level: string
    confidence: number
  }
  summary: string
  details: {
    nlp?: any
    vision?: any
    profile?: any
    tabular?: any
  }
}

interface ProfileInputFormProps {
  onAnalysisComplete: (result: AnalysisResult) => void
}

export default function ProfileInputForm({ onAnalysisComplete }: ProfileInputFormProps) {
  const [formData, setFormData] = useState<ProfileData>({
    username: '',
    displayName: '',
    platform: 'instagram',
    bio: '',
    profileText: '',
    followerCount: 0,
    followingCount: 0,
    postCount: 0,
    accountAge: undefined,
    verified: false
  })

  const [errors, setErrors] = useState<Partial<Record<keyof ProfileData, string>>>({})
  const [isLoading, setIsLoading] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)

  // Utility function to convert file to base64
  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.readAsDataURL(file)
      reader.onload = () => {
        if (typeof reader.result === 'string') {
          // Remove the data:image/jpeg;base64, prefix
          const base64 = reader.result.split(',')[1]
          resolve(base64)
        } else {
          reject(new Error('Failed to read file'))
        }
      }
      reader.onerror = reject
    })
  }

  // Cleanup preview URL when component unmounts
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  const updateField = (field: keyof ProfileData, value: string | number | boolean | undefined | File) => {
    setFormData(prev => ({ ...prev, [field]: value }))
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: undefined }))
    }
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        setErrors(prev => ({ ...prev, profilePicture: 'Please select an image file' }))
        return
      }
      
      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        setErrors(prev => ({ ...prev, profilePicture: 'File size must be less than 10MB' }))
        return
      }
      
      updateField('profilePicture', file)
      
      // Create preview URL
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
      
      // Clear any existing errors
      setErrors(prev => ({ ...prev, profilePicture: undefined }))
    }
  }

  const removeProfilePicture = () => {
    updateField('profilePicture', undefined)
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
      setPreviewUrl(null)
    }
    // Reset file input
    const fileInput = document.getElementById('profilePicture') as HTMLInputElement
    if (fileInput) {
      fileInput.value = ''
    }
  }

  const validateForm = (): boolean => {
    const newErrors: Partial<Record<keyof ProfileData, string>> = {}

    if (!formData.username.trim()) {
      newErrors.username = 'Username is required'
    }

    if (!formData.displayName.trim()) {
      newErrors.displayName = 'Display name is required'
    }

    if (!formData.platform) {
      newErrors.platform = 'Platform is required'
    }

    if (!formData.bio.trim()) {
      newErrors.bio = 'Bio is required'
    }

    if (!formData.profileText.trim()) {
      newErrors.profileText = 'Profile text content is required'
    }

    if (formData.followerCount < 0) {
      newErrors.followerCount = 'Follower count cannot be negative'
    }

    if (formData.followingCount < 0) {
      newErrors.followingCount = 'Following count cannot be negative'
    }

    if (formData.postCount < 0) {
      newErrors.postCount = 'Post count cannot be negative'
    }

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

    setIsLoading(true)

    try {
      // Prepare the request payload
      const payload: any = {
        type: 'manual',
        profileData: {
          profileUrl: `${formData.platform}.com/${formData.username}`, // Synthetic URL for API compatibility
          platform: formData.platform,
          username: formData.username,
          displayName: formData.displayName,
          bio: formData.bio,
          followerCount: formData.followerCount,
          followingCount: formData.followingCount,
          postCount: formData.postCount,
          accountAge: formData.accountAge,
          verified: formData.verified,
          posts: [], // No post analysis for manual input
          profileText: formData.profileText
        }
      }

      // Add profile picture if provided
      if (formData.profilePicture) {
        // Convert file to base64 for API
        const base64 = await fileToBase64(formData.profilePicture)
        payload.fileData = {
          name: formData.profilePicture.name,
          type: formData.profilePicture.type,
          size: formData.profilePicture.size,
          content: base64
        }
      }

      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      })

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.status} ${response.statusText}`)
      }

      const result: AnalysisResult = await response.json()
      onAnalysisComplete(result)

    } catch (error) {
      console.error('Analysis error:', error)
      // You might want to show an error message to the user here
      alert(`Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <User className="h-6 w-6 text-blue-600" />
          Manual Profile Analysis
        </CardTitle>
        <CardDescription>
          Enter profile information manually for AI-powered fake profile detection analysis
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Basic Profile Information */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="username">Username *</Label>
              <Input
                id="username"
                placeholder="e.g., john_doe, @username"
                value={formData.username}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateField('username', e.target.value)}
                disabled={isLoading}
                className={errors.username ? 'border-red-500' : ''}
              />
              {errors.username && <p className="text-sm text-red-500">{errors.username}</p>}
            </div>

            <div className="space-y-2">
              <Label htmlFor="displayName">Display Name *</Label>
              <Input
                id="displayName"
                placeholder="e.g., John Doe"
                value={formData.displayName}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateField('displayName', e.target.value)}
                disabled={isLoading}
                className={errors.displayName ? 'border-red-500' : ''}
              />
              {errors.displayName && <p className="text-sm text-red-500">{errors.displayName}</p>}
            </div>
          </div>

          {/* Platform Selection */}
          <div className="space-y-2">
            <Label htmlFor="platform">Platform *</Label>
            <select 
              value={formData.platform} 
              onChange={(e: React.ChangeEvent<HTMLSelectElement>) => updateField('platform', e.target.value)}
              className={`w-full px-3 py-2 border rounded-md ${errors.platform ? 'border-red-500' : 'border-gray-300'}`}
              disabled={isLoading}
            >
              <option value="">Select social media platform</option>
              <option value="instagram">Instagram</option>
              <option value="twitter">Twitter/X</option>
              <option value="facebook">Facebook</option>
              <option value="linkedin">LinkedIn</option>
              <option value="tiktok">TikTok</option>
              <option value="other">Other</option>
            </select>
            {errors.platform && <p className="text-sm text-red-500">{errors.platform}</p>}
          </div>

          {/* Profile Picture Upload */}
          <div className="space-y-2">
            <Label htmlFor="profilePicture">Profile Picture (Optional)</Label>
            <div className="space-y-3">
              {!formData.profilePicture ? (
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-gray-400 transition-colors">
                  <input
                    id="profilePicture"
                    type="file"
                    accept="image/*"
                    onChange={handleFileUpload}
                    disabled={isLoading}
                    className="hidden"
                  />
                  <label
                    htmlFor="profilePicture"
                    className="cursor-pointer flex flex-col items-center gap-2"
                  >
                    <Camera className="h-8 w-8 text-gray-400" />
                    <div className="text-sm text-gray-600">
                      <span className="font-medium text-blue-600 hover:text-blue-500">
                        Click to upload profile picture
                      </span>
                      <p className="text-gray-500 mt-1">PNG, JPG, GIF up to 10MB</p>
                    </div>
                  </label>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="flex items-center gap-4 p-4 border rounded-lg bg-gray-50">
                    {previewUrl && (
                      <img
                        src={previewUrl}
                        alt="Profile preview"
                        className="w-16 h-16 rounded-full object-cover border-2 border-gray-300"
                      />
                    )}
                    <div className="flex-1">
                      <p className="text-sm font-medium">{formData.profilePicture.name}</p>
                      <p className="text-xs text-gray-500">
                        {(formData.profilePicture.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={removeProfilePicture}
                      disabled={isLoading}
                      className="text-red-600 hover:text-red-700"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Profile picture will be analyzed for authenticity and manipulation detection
                  </p>
                </div>
              )}
            </div>
            {errors.profilePicture && <p className="text-sm text-red-500">{errors.profilePicture}</p>}
          </div>

          {/* Bio and Profile Text */}
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="bio">Bio/Description *</Label>
              <textarea
                id="bio"
                placeholder="Enter the profile bio or description..."
                value={formData.bio}
                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => updateField('bio', e.target.value)}
                disabled={isLoading}
                rows={3}
                className={`w-full px-3 py-2 border rounded-md ${errors.bio ? 'border-red-500' : 'border-gray-300'}`}
              />
              {errors.bio && <p className="text-sm text-red-500">{errors.bio}</p>}
            </div>

            <div className="space-y-2">
              <Label htmlFor="profileText">Profile Text Content *</Label>
              <textarea
                id="profileText"
                placeholder="Enter any text content from posts, captions, or other profile text..."
                value={formData.profileText}
                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => updateField('profileText', e.target.value)}
                disabled={isLoading}
                rows={4}
                className={`w-full px-3 py-2 border rounded-md ${errors.profileText ? 'border-red-500' : 'border-gray-300'}`}
              />
              {errors.profileText && <p className="text-sm text-red-500">{errors.profileText}</p>}
              <p className="text-sm text-muted-foreground">
                This text will be analyzed for sentiment, authenticity, and other linguistic patterns
              </p>
            </div>
          </div>

          {/* Profile Metrics */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="h-5 w-5 text-blue-600" />
              <h3 className="text-lg font-semibold">Profile Metrics</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label htmlFor="followerCount">Followers *</Label>
                <Input
                  id="followerCount"
                  type="number"
                  min="0"
                  placeholder="0"
                  value={formData.followerCount || ''}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateField('followerCount', parseInt(e.target.value) || 0)}
                  disabled={isLoading}
                  className={errors.followerCount ? 'border-red-500' : ''}
                />
                {errors.followerCount && <p className="text-sm text-red-500">{errors.followerCount}</p>}
              </div>

              <div className="space-y-2">
                <Label htmlFor="followingCount">Following *</Label>
                <Input
                  id="followingCount"
                  type="number"
                  min="0"
                  placeholder="0"
                  value={formData.followingCount || ''}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateField('followingCount', parseInt(e.target.value) || 0)}
                  disabled={isLoading}
                  className={errors.followingCount ? 'border-red-500' : ''}
                />
                {errors.followingCount && <p className="text-sm text-red-500">{errors.followingCount}</p>}
              </div>

              <div className="space-y-2">
                <Label htmlFor="postCount">Posts *</Label>
                <Input
                  id="postCount"
                  type="number"
                  min="0"
                  placeholder="0"
                  value={formData.postCount || ''}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateField('postCount', parseInt(e.target.value) || 0)}
                  disabled={isLoading}
                  className={errors.postCount ? 'border-red-500' : ''}
                />
                {errors.postCount && <p className="text-sm text-red-500">{errors.postCount}</p>}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="accountAge">Account Age (days)</Label>
                <Input
                  id="accountAge"
                  type="number"
                  min="0"
                  placeholder="Optional - leave blank if unknown"
                  value={formData.accountAge || ''}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateField('accountAge', e.target.value ? parseInt(e.target.value) : undefined)}
                  disabled={isLoading}
                  className={errors.accountAge ? 'border-red-500' : ''}
                />
                {errors.accountAge && <p className="text-sm text-red-500">{errors.accountAge}</p>}
                <p className="text-sm text-muted-foreground">Optional: How many days old is this account?</p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="verified">Verification Status</Label>
                <select 
                  value={formData.verified ? 'true' : 'false'} 
                  onChange={(e: React.ChangeEvent<HTMLSelectElement>) => updateField('verified', e.target.value === 'true')}
                  className="w-full px-3 py-2 border rounded-md border-gray-300"
                  disabled={isLoading}
                >
                  <option value="false">Not Verified</option>
                  <option value="true">Verified Account</option>
                </select>
              </div>
            </div>
          </div>

          {/* Submit Button */}
          <div className="pt-4">
            <Button 
              type="submit" 
              className="w-full" 
              disabled={isLoading}
              size="lg"
            >
              {isLoading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                  Analyzing Profile...
                </>
              ) : (
                <>
                  <BarChart3 className="h-4 w-4 mr-2" />
                  Analyze Profile
                </>
              )}
            </Button>
          </div>

          {/* Required Fields Note */}
          <p className="text-sm text-muted-foreground text-center">
            * Required fields. All data is processed locally for analysis.
          </p>
        </form>
      </CardContent>
    </Card>
  )
}