'use client'

import { useState } from 'react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Link2 } from 'lucide-react'

export interface ExtractedProfileData {
  platform: string
  username: string
  displayName: string
  bio: string
  followerCount: number
  followingCount: number
  postCount: number
  profileImageUrl: string
  manualInputRequired: boolean
  missingFields: string[]
  extractionError?: string
}

interface UrlExtractFormProps {
  onExtracted: (data: ExtractedProfileData) => void
}

export function UrlExtractForm({ onExtracted }: UrlExtractFormProps) {
  const [url, setUrl] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!url.trim()) return

    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch('/api/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim() })
      })
      const data: ExtractedProfileData = await response.json()
      onExtracted(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Extraction failed')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-3">
      <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row gap-3">
        <Input
          type="url"
          placeholder="https://instagram.com/username or https://x.com/username"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={isLoading}
          className="flex-1"
        />
        <Button type="submit" disabled={isLoading || !url.trim()} className="flex items-center gap-2">
          <Link2 className="h-4 w-4" />
          {isLoading ? 'Extracting...' : 'Extract & Continue'}
        </Button>
      </form>
      {error && <p className="text-sm text-red-500">{error}</p>}
      <p className="text-xs text-muted-foreground">
        Best-effort extraction from public profile pages. Instagram/X frequently block or limit this - whatever
        can&apos;t be extracted automatically, you&apos;ll be asked to fill in on the next step.
      </p>
    </div>
  )
}
