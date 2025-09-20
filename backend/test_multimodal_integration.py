#!/usr/bin/env python3

import requests
import json
import time
import asyncio
import concurrent.futures
from typing import Dict, List, Any
import statistics
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TestResult:
    """Test result container."""
    service_name: str
    endpoint: str
    success: bool
    response_time_ms: float
    status_code: int
    error_message: str = ""
    response_data: Dict = None

class MultiModalProfileDetector:
    """
    Integration client for all four detection services.
    """
    
    def __init__(self):
        self.services = {
            "text_analysis": "http://127.0.0.1:8000",
            "ml_classification": "http://127.0.0.1:8001", 
            "vision_detection": "http://127.0.0.1:8002",
            "tabular_classification": "http://127.0.0.1:8003",
            "ensemble_tabular": "http://127.0.0.1:8004"
        }
        self.test_results = []
    
    def check_service_health(self, service_name: str, base_url: str) -> TestResult:
        """Check if a service is healthy."""
        try:
            start_time = time.time()
            response = requests.get(f"{base_url}/health", timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            return TestResult(
                service_name=service_name,
                endpoint="/health",
                success=response.status_code == 200,
                response_time_ms=response_time,
                status_code=response.status_code,
                response_data=response.json() if response.status_code == 200 else None
            )
        except Exception as e:
            return TestResult(
                service_name=service_name,
                endpoint="/health",
                success=False,
                response_time_ms=0,
                status_code=0,
                error_message=str(e)
            )
    
    def test_text_analysis(self, text: str) -> TestResult:
        """Test text analysis service."""
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.services['text_analysis']}/analyze",
                json={"text": text},
                timeout=15
            )
            response_time = (time.time() - start_time) * 1000
            
            return TestResult(
                service_name="text_analysis",
                endpoint="/analyze",
                success=response.status_code == 200,
                response_time_ms=response_time,
                status_code=response.status_code,
                response_data=response.json() if response.status_code == 200 else None
            )
        except Exception as e:
            return TestResult(
                service_name="text_analysis",
                endpoint="/analyze",
                success=False,
                response_time_ms=0,
                status_code=0,
                error_message=str(e)
            )
    
    def test_vision_detection(self, image_url: str) -> TestResult:
        """Test vision detection service."""
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.services['vision_detection']}/detect",
                json={"image_url": image_url, "model_type": "resnet50"},
                timeout=20
            )
            response_time = (time.time() - start_time) * 1000
            
            return TestResult(
                service_name="vision_detection",
                endpoint="/detect",
                success=response.status_code == 200,
                response_time_ms=response_time,
                status_code=response.status_code,
                response_data=response.json() if response.status_code == 200 else None
            )
        except Exception as e:
            return TestResult(
                service_name="vision_detection",
                endpoint="/detect",
                success=False,
                response_time_ms=0,
                status_code=0,
                error_message=str(e)
            )
    
    def test_tabular_classification(self, features: Dict[str, float], service_type: str = "simple") -> TestResult:
        """Test tabular classification service."""
        service_key = "ensemble_tabular" if service_type == "ensemble" else "tabular_classification"
        base_url = self.services[service_key]
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/predict",
                json={"features": features},
                timeout=10
            )
            response_time = (time.time() - start_time) * 1000
            
            return TestResult(
                service_name=f"tabular_{service_type}",
                endpoint="/predict",
                success=response.status_code == 200,
                response_time_ms=response_time,
                status_code=response.status_code,
                response_data=response.json() if response.status_code == 200 else None
            )
        except Exception as e:
            return TestResult(
                service_name=f"tabular_{service_type}",
                endpoint="/predict",
                success=False,
                response_time_ms=0,
                status_code=0,
                error_message=str(e)
            )
    
    def analyze_complete_profile(self, profile_data: Dict[str, Any]) -> Dict[str, TestResult]:
        """
        Analyze a complete profile using all available services.
        
        Args:
            profile_data: Dictionary containing:
                - text: Profile text/bio
                - image_url: Profile image URL
                - features: Behavioral features dict
        
        Returns:
            Dictionary of test results from each service
        """
        results = {}
        
        # Test text analysis
        if "text" in profile_data:
            results["text"] = self.test_text_analysis(profile_data["text"])
        
        # Test vision detection
        if "image_url" in profile_data:
            results["vision"] = self.test_vision_detection(profile_data["image_url"])
        
        # Test tabular classification (both simple and ensemble)
        if "features" in profile_data:
            results["tabular_simple"] = self.test_tabular_classification(
                profile_data["features"], "simple"
            )
            results["tabular_ensemble"] = self.test_tabular_classification(
                profile_data["features"], "ensemble"
            )
        
        return results
    
    def calculate_ensemble_score(self, results: Dict[str, TestResult]) -> Dict[str, Any]:
        """
        Calculate an ensemble score from all successful predictions.
        """
        scores = []
        weights = {
            "text": 0.25,
            "vision": 0.35,
            "tabular_simple": 0.20,
            "tabular_ensemble": 0.20
        }
        
        ensemble_data = {
            "individual_scores": {},
            "successful_services": [],
            "failed_services": [],
            "total_response_time_ms": 0
        }
        
        for service, result in results.items():
            ensemble_data["total_response_time_ms"] += result.response_time_ms
            
            if result.success and result.response_data:
                ensemble_data["successful_services"].append(service)
                
                # Extract probability score based on service type
                if service == "text":
                    # Text analysis returns sentiment score
                    sentiment = result.response_data.get("sentiment", {})
                    score = sentiment.get("positive", 0.5)  # Use positive sentiment as real indicator
                elif service == "vision":
                    # Vision returns probability of being real
                    score = result.response_data.get("probability_real", 0.5)
                elif service.startswith("tabular"):
                    # Tabular returns probability of being real
                    score = result.response_data.get("probability_real", 0.5)
                else:
                    score = 0.5
                
                ensemble_data["individual_scores"][service] = score
                
                # Add to weighted ensemble if weight exists
                if service in weights:
                    scores.append((score, weights[service]))
            else:
                ensemble_data["failed_services"].append(service)
        
        # Calculate weighted ensemble score
        if scores:
            weighted_sum = sum(score * weight for score, weight in scores)
            total_weight = sum(weight for _, weight in scores)
            ensemble_score = weighted_sum / total_weight
        else:
            ensemble_score = 0.5  # Neutral if no services succeeded
        
        ensemble_data["ensemble_score"] = ensemble_score
        ensemble_data["classification"] = "real" if ensemble_score > 0.5 else "fake"
        ensemble_data["confidence"] = "high" if abs(ensemble_score - 0.5) > 0.3 else "medium" if abs(ensemble_score - 0.5) > 0.15 else "low"
        
        return ensemble_data
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test of all services."""
print("Starting Multi-Modal Integration Test")
print("=" * 60)        # Test service health
        print("\n💓 Health Check Results:")
        health_results = {}
        for service_name, base_url in self.services.items():
            result = self.check_service_health(service_name, base_url)
            health_results[service_name] = result
            status = "✅" if result.success else "❌"
            print(f"  {status} {service_name}: {result.response_time_ms:.1f}ms")
        
        # Test profile scenarios
        test_profiles = [
            {
                "name": "Real User Profile",
                "text": "Software engineer at tech company. Love hiking and photography. Based in San Francisco.",
                "image_url": "https://example.com/real_profile.jpg",
                "features": {
                    "account_age_days": 1095,
                    "followers_following_ratio": 1.8,
                    "post_frequency": 0.15,
                    "engagement_per_post": 22.0
                }
            },
            {
                "name": "Suspected Bot Profile", 
                "text": "Follow me for amazing content! Check out my link in bio for exclusive deals!",
                "image_url": "https://example.com/bot_profile.jpg",
                "features": {
                    "account_age_days": 5,
                    "followers_following_ratio": 0.005,
                    "post_frequency": 25.0,
                    "engagement_per_post": 0.2
                }
            },
            {
                "name": "Suspicious Celebrity Clone",
                "text": "Official account of famous celebrity. DM for collaborations.",
                "image_url": "https://example.com/fake_celebrity.jpg", 
                "features": {
                    "account_age_days": 15,
                    "followers_following_ratio": 100.0,
                    "post_frequency": 8.0,
                    "engagement_per_post": 2.5
                }
            }
        ]
        
        print(f"\n🎭 Testing {len(test_profiles)} Profile Scenarios:")
        scenario_results = {}
        
        for i, profile in enumerate(test_profiles):
            print(f"\n  📊 Scenario {i+1}: {profile['name']}")
            
            # Analyze complete profile
            results = self.analyze_complete_profile(profile)
            
            # Calculate ensemble score
            ensemble_data = self.calculate_ensemble_score(results)
            
            # Display results
            print(f"    Services: {len(ensemble_data['successful_services'])}/{len(results)} successful")
            print(f"    Ensemble Score: {ensemble_data['ensemble_score']:.3f}")
            print(f"    Classification: {ensemble_data['classification']} ({ensemble_data['confidence']} confidence)")
            print(f"    Total Time: {ensemble_data['total_response_time_ms']:.1f}ms")
            
            if ensemble_data['individual_scores']:
                print("    Individual Scores:")
                for service, score in ensemble_data['individual_scores'].items():
                    print(f"      {service}: {score:.3f}")
            
            scenario_results[profile['name']] = {
                "results": results,
                "ensemble": ensemble_data
            }
        
        # Performance analysis
        print(f"\n📈 Performance Analysis:")
        all_response_times = []
        service_performance = {}
        
        for scenario_data in scenario_results.values():
            for service, result in scenario_data["results"].items():
                if result.success:
                    all_response_times.append(result.response_time_ms)
                    if service not in service_performance:
                        service_performance[service] = []
                    service_performance[service].append(result.response_time_ms)
        
        if all_response_times:
            print(f"  Overall average response time: {statistics.mean(all_response_times):.1f}ms")
            print(f"  Fastest response: {min(all_response_times):.1f}ms")
            print(f"  Slowest response: {max(all_response_times):.1f}ms")
            
            print(f"\n  Service Performance:")
            for service, times in service_performance.items():
                avg_time = statistics.mean(times)
                print(f"    {service}: {avg_time:.1f}ms avg ({len(times)} requests)")
        
        # System recommendations
        print(f"\n💡 System Recommendations:")
        working_services = sum(1 for result in health_results.values() if result.success)
        total_services = len(health_results)
        
        if working_services == total_services:
            print(f"  ✅ All {total_services} services operational - full multi-modal detection available")
        elif working_services >= 3:
            print(f"  ⚠️  {working_services}/{total_services} services working - degraded but functional")
        else:
            print(f"  ❌ Only {working_services}/{total_services} services working - limited functionality")
        
        # Integration success metrics
        successful_integrations = sum(
            1 for data in scenario_results.values() 
            if len(data["ensemble"]["successful_services"]) >= 2
        )
        
        print(f"  🔗 Multi-modal integration: {successful_integrations}/{len(test_profiles)} scenarios successful")
        
        return {
            "health_results": health_results,
            "scenario_results": scenario_results,
            "performance": {
                "service_performance": service_performance,
                "overall_stats": {
                    "total_requests": len(all_response_times),
                    "average_response_ms": statistics.mean(all_response_times) if all_response_times else 0,
                    "working_services": working_services,
                    "total_services": total_services
                }
            },
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Run the comprehensive integration test."""
    detector = MultiModalProfileDetector()
    results = detector.run_comprehensive_test()
    
    print(f"\n🎉 Integration Test Complete!")
    print(f"📄 Full results available in test output")
    
    # Optional: Save results to file
    with open(f"integration_test_results_{int(time.time())}.json", "w") as f:
        # Convert TestResult objects to dicts for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if key == "health_results" or key == "scenario_results":
                serializable_value = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict) and "results" in subvalue:
                        # Handle scenario results
                        serializable_subvalue = {
                            "ensemble": subvalue["ensemble"],
                            "results": {
                                k: {
                                    "service_name": v.service_name,
                                    "endpoint": v.endpoint,
                                    "success": v.success,
                                    "response_time_ms": v.response_time_ms,
                                    "status_code": v.status_code,
                                    "error_message": v.error_message,
                                    "response_data": v.response_data
                                } for k, v in subvalue["results"].items()
                            }
                        }
                        serializable_value[subkey] = serializable_subvalue
                    elif hasattr(subvalue, "service_name"):
                        # Handle TestResult objects
                        serializable_value[subkey] = {
                            "service_name": subvalue.service_name,
                            "endpoint": subvalue.endpoint,
                            "success": subvalue.success,
                            "response_time_ms": subvalue.response_time_ms,
                            "status_code": subvalue.status_code,
                            "error_message": subvalue.error_message,
                            "response_data": subvalue.response_data
                        }
                    else:
                        serializable_value[subkey] = subvalue
                serializable_results[key] = serializable_value
            else:
                serializable_results[key] = value
        
        json.dump(serializable_results, f, indent=2)

if __name__ == "__main__":
    main()