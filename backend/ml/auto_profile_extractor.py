"""
Profile Purity Detector - Automatic Profile Data Extraction
Free methods to extract profile information from Instagram and Twitter links
Uses web scraping, public APIs, and metadata extraction
"""

import requests
import re
import json
import time
import logging
from typing import Dict, Optional, Any
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoProfileExtractor:
    """
    Automatic profile data extraction for Profile Purity Detector
    Extracts follower count, following count, posts, bio, profile image, etc.
    """
    
    def __init__(self):
        self.session = requests.Session()
        # Use realistic headers to avoid being blocked
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def extract_profile_data(self, url: str) -> Dict[str, Any]:
        """
        Main method to extract profile data from any supported platform
        
        Args:
            url: Profile URL (Instagram, Twitter, etc.)
            
        Returns:
            Dictionary with extracted profile data
        """
        try:
            platform = self._detect_platform(url)
            logger.info(f"Extracting data from {platform} profile: {url}")
            
            if platform == 'instagram':
                return self._extract_instagram_data(url)
            elif platform == 'twitter':
                return self._extract_twitter_data(url)
            elif platform == 'facebook':
                return self._extract_facebook_data(url)
            else:
                raise ValueError(f"Unsupported platform: {platform}")
                
        except Exception as e:
            logger.error(f"Error extracting profile data: {e}")
            return self._get_fallback_data(url)
    
    def _detect_platform(self, url: str) -> str:
        """Detect social media platform from URL"""
        url_lower = url.lower()
        if 'instagram.com' in url_lower:
            return 'instagram'
        elif 'twitter.com' in url_lower or 'x.com' in url_lower:
            return 'twitter'
        elif 'facebook.com' in url_lower:
            return 'facebook'
        else:
            return 'unknown'
    
    def _extract_instagram_data(self, url: str) -> Dict[str, Any]:
        """
        Extract Instagram profile data using free methods
        """
        try:
            # Method 1: Try Instagram's public JSON endpoint
            username = self._extract_username_from_url(url)
            
            # Instagram public endpoint (sometimes available)
            api_url = f"https://www.instagram.com/{username}/?__a=1&__d=dis"
            
            try:
                response = self.session.get(api_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return self._parse_instagram_json(data, username)
            except:
                pass
            
            # Method 2: Scrape public Instagram page
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                return self._scrape_instagram_html(response.text, username)
            
        except Exception as e:
            logger.error(f"Instagram extraction error: {e}")
            
        return self._get_fallback_data(url, 'instagram')
    
    def _extract_twitter_data(self, url: str) -> Dict[str, Any]:
        """
        Extract Twitter profile data using free methods
        """
        try:
            username = self._extract_username_from_url(url)
            
            # Method 1: Try Twitter's guest token approach (free)
            guest_token = self._get_twitter_guest_token()
            if guest_token:
                profile_data = self._fetch_twitter_profile_api(username, guest_token)
                if profile_data:
                    return profile_data
            
            # Method 2: Scrape Twitter profile page
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                return self._scrape_twitter_html(response.text, username)
                
        except Exception as e:
            logger.error(f"Twitter extraction error: {e}")
            
        return self._get_fallback_data(url, 'twitter')
    
    def _parse_instagram_json(self, data: Dict, username: str) -> Dict[str, Any]:
        """Parse Instagram JSON response"""
        try:
            user_data = data.get('graphql', {}).get('user', {})
            
            return {
                'platform': 'instagram',
                'username': username,
                'displayName': user_data.get('full_name', ''),
                'bio': user_data.get('biography', ''),
                'followerCount': user_data.get('edge_followed_by', {}).get('count', 0),
                'followingCount': user_data.get('edge_follow', {}).get('count', 0),
                'postCount': user_data.get('edge_owner_to_timeline_media', {}).get('count', 0),
                'verified': user_data.get('is_verified', False),
                'profileImageUrl': user_data.get('profile_pic_url_hd', user_data.get('profile_pic_url', '')),
                'isPrivate': user_data.get('is_private', False),
                'externalUrl': user_data.get('external_url', ''),
                'accountAge': self._estimate_account_age(user_data.get('date_joined')),
                'extractionMethod': 'instagram_api'
            }
            
        except Exception as e:
            logger.error(f"Error parsing Instagram JSON: {e}")
            return self._get_fallback_data(f"https://instagram.com/{username}", 'instagram')
    
    def _scrape_instagram_html(self, html: str, username: str) -> Dict[str, Any]:
        """Scrape Instagram profile from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for JSON data in script tags
            script_tags = soup.find_all('script', type='text/javascript')
            profile_data = {}
            
            for script in script_tags:
                if script.string and 'window._sharedData' in script.string:
                    # Extract shared data
                    json_str = script.string.split('window._sharedData = ')[1].split(';</script>')[0]
                    shared_data = json.loads(json_str)
                    
                    entry_data = shared_data.get('entry_data', {})
                    profile_page = entry_data.get('ProfilePage', [{}])[0]
                    user_data = profile_page.get('graphql', {}).get('user', {})
                    
                    if user_data:
                        return self._parse_instagram_json({'graphql': {'user': user_data}}, username)
            
            # Fallback: Extract from meta tags
            follower_count = self._extract_meta_content(soup, ['followers', 'Followers'])
            following_count = self._extract_meta_content(soup, ['following', 'Following'])
            post_count = self._extract_meta_content(soup, ['posts', 'Posts'])
            
            return {
                'platform': 'instagram',
                'username': username,
                'displayName': soup.find('meta', property='og:title')['content'].split(' (')[0] if soup.find('meta', property='og:title') else '',
                'bio': soup.find('meta', property='og:description')['content'] if soup.find('meta', property='og:description') else '',
                'followerCount': self._parse_count_string(follower_count),
                'followingCount': self._parse_count_string(following_count),
                'postCount': self._parse_count_string(post_count),
                'verified': 'verified' in html.lower(),
                'profileImageUrl': soup.find('meta', property='og:image')['content'] if soup.find('meta', property='og:image') else '',
                'accountAge': None,
                'extractionMethod': 'instagram_scraping'
            }
            
        except Exception as e:
            logger.error(f"Error scraping Instagram HTML: {e}")
            return self._get_fallback_data(f"https://instagram.com/{username}", 'instagram')
    
    def _get_twitter_guest_token(self) -> Optional[str]:
        """Get Twitter guest token for API access"""
        try:
            response = self.session.post(
                'https://api.twitter.com/1.1/guest/activate.json',
                headers={'Authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA'}
            )
            if response.status_code == 200:
                return response.json().get('guest_token')
        except Exception as e:
            logger.error(f"Error getting Twitter guest token: {e}")
        return None
    
    def _fetch_twitter_profile_api(self, username: str, guest_token: str) -> Optional[Dict[str, Any]]:
        """Fetch Twitter profile using guest token"""
        try:
            headers = {
                'Authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA',
                'x-guest-token': guest_token
            }
            
            # Twitter API v1.1 user lookup
            response = self.session.get(
                f'https://api.twitter.com/1.1/users/show.json?screen_name={username}',
                headers=headers
            )
            
            if response.status_code == 200:
                user_data = response.json()
                return {
                    'platform': 'twitter',
                    'username': username,
                    'displayName': user_data.get('name', ''),
                    'bio': user_data.get('description', ''),
                    'followerCount': user_data.get('followers_count', 0),
                    'followingCount': user_data.get('friends_count', 0),
                    'postCount': user_data.get('statuses_count', 0),
                    'verified': user_data.get('verified', False),
                    'profileImageUrl': user_data.get('profile_image_url_https', '').replace('_normal', '_400x400'),
                    'accountAge': self._calculate_twitter_account_age(user_data.get('created_at')),
                    'location': user_data.get('location', ''),
                    'extractionMethod': 'twitter_api'
                }
                
        except Exception as e:
            logger.error(f"Error fetching Twitter profile API: {e}")
        return None
    
    def _scrape_twitter_html(self, html: str, username: str) -> Dict[str, Any]:
        """Scrape Twitter profile from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract from meta tags
            title = soup.find('meta', property='og:title')
            description = soup.find('meta', property='og:description')
            image = soup.find('meta', property='og:image')
            
            # Look for JSON data in script tags
            script_tags = soup.find_all('script')
            for script in script_tags:
                if script.string and 'profile' in script.string.lower():
                    # Try to extract profile data from script content
                    pass
            
            return {
                'platform': 'twitter',
                'username': username,
                'displayName': title['content'].split(' (')[0] if title else '',
                'bio': description['content'] if description else '',
                'followerCount': 0,  # Hard to extract without API
                'followingCount': 0,
                'postCount': 0,
                'verified': 'verified' in html.lower(),
                'profileImageUrl': image['content'] if image else '',
                'accountAge': None,
                'extractionMethod': 'twitter_scraping'
            }
            
        except Exception as e:
            logger.error(f"Error scraping Twitter HTML: {e}")
            return self._get_fallback_data(f"https://twitter.com/{username}", 'twitter')
    
    def _extract_facebook_data(self, url: str) -> Dict[str, Any]:
        """Extract Facebook profile data (limited due to privacy)"""
        username = self._extract_username_from_url(url)
        
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                return {
                    'platform': 'facebook',
                    'username': username,
                    'displayName': soup.find('meta', property='og:title')['content'] if soup.find('meta', property='og:title') else '',
                    'bio': soup.find('meta', property='og:description')['content'] if soup.find('meta', property='og:description') else '',
                    'followerCount': 0,  # Facebook doesn't expose this
                    'followingCount': 0,
                    'postCount': 0,
                    'verified': False,
                    'profileImageUrl': soup.find('meta', property='og:image')['content'] if soup.find('meta', property='og:image') else '',
                    'accountAge': None,
                    'extractionMethod': 'facebook_scraping'
                }
        except Exception as e:
            logger.error(f"Facebook extraction error: {e}")
            
        return self._get_fallback_data(url, 'facebook')
    
    def _extract_username_from_url(self, url: str) -> str:
        """Extract username from profile URL"""
        # Remove trailing slash and extract last part
        path = urlparse(url).path.strip('/')
        parts = path.split('/')
        
        # Handle different URL formats
        if len(parts) >= 1:
            username = parts[-1]
            # Remove query parameters
            username = username.split('?')[0]
            return username
        
        return 'unknown'
    
    def _extract_meta_content(self, soup: BeautifulSoup, keywords: list) -> str:
        """Extract content from meta tags or text containing keywords"""
        for keyword in keywords:
            # Try meta tags first
            meta = soup.find('meta', attrs={'name': lambda x: x and keyword.lower() in x.lower()})
            if meta and meta.get('content'):
                return meta['content']
            
            # Try text content
            text_elements = soup.find_all(text=re.compile(keyword, re.IGNORECASE))
            for element in text_elements:
                # Look for numbers near the keyword
                numbers = re.findall(r'[\d,\.]+', element)
                if numbers:
                    return numbers[0]
        
        return '0'
    
    def _parse_count_string(self, count_str: str) -> int:
        """Parse count strings like '1.2M', '1,234', etc."""
        if not count_str:
            return 0
            
        # Remove commas and spaces
        count_str = count_str.replace(',', '').replace(' ', '').upper()
        
        # Handle K, M, B suffixes
        if 'K' in count_str:
            return int(float(count_str.replace('K', '')) * 1000)
        elif 'M' in count_str:
            return int(float(count_str.replace('M', '')) * 1000000)
        elif 'B' in count_str:
            return int(float(count_str.replace('B', '')) * 1000000000)
        else:
            try:
                return int(count_str)
            except:
                return 0
    
    def _estimate_account_age(self, date_string: Optional[str]) -> Optional[int]:
        """Estimate account age in days"""
        if not date_string:
            return None
            
        try:
            # Parse different date formats
            from datetime import datetime
            
            # Try common formats
            formats = ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%a %b %d %H:%M:%S %z %Y']
            
            for fmt in formats:
                try:
                    date = datetime.strptime(date_string, fmt)
                    age_days = (datetime.now() - date).days
                    return max(0, age_days)
                except:
                    continue
                    
        except Exception as e:
            logger.error(f"Error estimating account age: {e}")
        
        return None
    
    def _calculate_twitter_account_age(self, created_at: str) -> Optional[int]:
        """Calculate Twitter account age from created_at string"""
        try:
            from datetime import datetime
            # Twitter format: "Wed Oct 10 20:19:24 +0000 2018"
            date = datetime.strptime(created_at, '%a %b %d %H:%M:%S %z %Y')
            age_days = (datetime.now().replace(tzinfo=date.tzinfo) - date).days
            return max(0, age_days)
        except Exception as e:
            logger.error(f"Error calculating Twitter account age: {e}")
            return None
    
    def _get_fallback_data(self, url: str, platform: str = None) -> Dict[str, Any]:
        """Return fallback data when extraction fails"""
        username = self._extract_username_from_url(url)
        detected_platform = platform or self._detect_platform(url)
        
        return {
            'platform': detected_platform,
            'username': username,
            'displayName': '',
            'bio': '',
            'followerCount': 0,
            'followingCount': 0,
            'postCount': 0,
            'verified': False,
            'profileImageUrl': '',
            'accountAge': None,
            'extractionMethod': 'fallback',
            'extractionError': 'Could not extract profile data automatically',
            'manualInputRequired': True
        }

# Usage example
if __name__ == "__main__":
    extractor = AutoProfileExtractor()
    
    # Test URLs
    test_urls = [
        "https://instagram.com/cristiano",
        "https://twitter.com/elonmusk",
        "https://www.instagram.com/selenagomez/",
        "https://x.com/taylorswift13"
    ]
    
    for url in test_urls:
        print(f"\n🔍 Extracting data from: {url}")
        data = extractor.extract_profile_data(url)
        print(json.dumps(data, indent=2))