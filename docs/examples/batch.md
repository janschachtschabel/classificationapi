# Batch Processing Examples

Examples of processing multiple texts efficiently using the Classification API.

## Async Batch Processing

### Python with asyncio and aiohttp

```python
import asyncio
import aiohttp
from typing import List, Dict, Any
import time

class BatchClassifier:
    """Efficient batch processing for the Classification API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    async def classify_single(
        self, 
        session: aiohttp.ClientSession, 
        text: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """Classify a single text asynchronously."""
        
        data = {
            "text": text,
            "mode": kwargs.get("mode", "skos"),
            **kwargs
        }
        
        try:
            async with session.post(f"{self.base_url}/classify", json=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_data = await response.json()
                    return {
                        "error": f"HTTP {response.status}",
                        "detail": error_data.get("detail", "Unknown error")
                    }
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
    
    async def classify_batch(
        self, 
        texts: List[str], 
        batch_size: int = 5,
        delay_between_batches: float = 1.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process texts in batches to avoid rate limiting."""
        
        results = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Process batch concurrently
                tasks = [
                    self.classify_single(session, text, **kwargs) 
                    for text in batch
                ]
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                results.extend(batch_results)
                
                # Delay between batches to respect rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(delay_between_batches)
        
        return results

# Example usage
async def main():
    classifier = BatchClassifier()
    
    texts = [
        "Introduction to machine learning algorithms",
        "Photosynthesis in plant biology",
        "European history in the 19th century",
        "Quantum mechanics for beginners",
        "Database design principles",
        "Creative writing techniques",
        "Statistical analysis methods",
        "Environmental science concepts"
    ]
    
    start_time = time.time()
    
    results = await classifier.classify_batch(
        texts,
        mode="skos",
        generate_descriptive_fields=True,
        batch_size=3,
        delay_between_batches=0.5
    )
    
    end_time = time.time()
    
    print(f"Processed {len(texts)} texts in {end_time - start_time:.2f} seconds")
    
    for i, result in enumerate(results):
        if "error" not in result:
            title = result.get("descriptive_fields", {}).get("title", "N/A")
            print(f"Text {i+1}: {title}")
            
            for vocab_result in result.get("results", []):
                for classification in vocab_result.get("classifications", []):
                    print(f"  - {classification['preferred_label']} ({classification['confidence']:.2%})")
        else:
            print(f"Text {i+1}: Error - {result['error']}")

# Run the batch processing
if __name__ == "__main__":
    asyncio.run(main())
```

## Concurrent Processing with Rate Limiting

### Advanced Batch Processor

```python
import asyncio
import aiohttp
from asyncio import Semaphore
from typing import List, Dict, Any, Optional
import logging

class AdvancedBatchProcessor:
    """Advanced batch processor with rate limiting and error handling."""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000",
        max_concurrent: int = 5,
        rate_limit_per_minute: int = 60
    ):
        self.base_url = base_url
        self.semaphore = Semaphore(max_concurrent)
        self.rate_limit = rate_limit_per_minute
        self.request_times = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def _rate_limit_check(self):
        """Ensure we don't exceed rate limits."""
        now = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # If we're at the rate limit, wait
        if len(self.request_times) >= self.rate_limit:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        self.request_times.append(now)
    
    async def process_single_with_retry(
        self,
        session: aiohttp.ClientSession,
        text: str,
        max_retries: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """Process single text with retry logic."""
        
        async with self.semaphore:
            await self._rate_limit_check()
            
            for attempt in range(max_retries):
                try:
                    data = {
                        "text": text,
                        "mode": kwargs.get("mode", "skos"),
                        **kwargs
                    }
                    
                    async with session.post(
                        f"{self.base_url}/classify", 
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 429:  # Rate limited
                            wait_time = 2 ** attempt
                            self.logger.warning(f"Rate limited, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            error_data = await response.json()
                            return {
                                "error": f"HTTP {response.status}",
                                "detail": error_data.get("detail", "Unknown error"),
                                "text_preview": text[:100] + "..." if len(text) > 100 else text
                            }
                
                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout on attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        return {
                            "error": "Timeout",
                            "detail": "Request timed out after all retries",
                            "text_preview": text[:100] + "..." if len(text) > 100 else text
                        }
                    await asyncio.sleep(2 ** attempt)
                
                except Exception as e:
                    self.logger.error(f"Unexpected error: {e}")
                    return {
                        "error": "Exception",
                        "detail": str(e),
                        "text_preview": text[:100] + "..." if len(text) > 100 else text
                    }
        
        return {"error": "Max retries exceeded"}
    
    async def process_batch(
        self,
        texts: List[str],
        progress_callback: Optional[callable] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process batch with progress tracking."""
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.process_single_with_retry(session, text, **kwargs)
                for text in texts
            ]
            
            results = []
            completed = 0
            
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, len(texts))
                
                self.logger.info(f"Completed {completed}/{len(texts)} requests")
        
        return results

# Example with progress tracking
async def batch_with_progress():
    def progress_callback(completed: int, total: int):
        percentage = (completed / total) * 100
        print(f"Progress: {completed}/{total} ({percentage:.1f}%)")
    
    processor = AdvancedBatchProcessor(
        max_concurrent=3,
        rate_limit_per_minute=30
    )
    
    texts = [
        "Advanced calculus concepts for engineering students",
        "Introduction to organic chemistry reactions",
        "World War II historical analysis",
        "Machine learning in healthcare applications",
        "Environmental sustainability practices",
        "Creative writing workshop techniques",
        "Statistical modeling for business analytics",
        "Quantum physics theoretical foundations",
        "Database optimization strategies",
        "Art history Renaissance period"
    ]
    
    results = await processor.process_batch(
        texts,
        mode="skos",
        generate_descriptive_fields=True,
        resource_suggestion=True,
        progress_callback=progress_callback
    )
    
    # Analyze results
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    
    print(f"\nResults: {len(successful)} successful, {len(failed)} failed")
    
    if failed:
        print("\nFailed requests:")
        for i, failure in enumerate(failed):
            print(f"  {i+1}. {failure['error']}: {failure.get('detail', 'N/A')}")
    
    return results

# Run advanced batch processing
if __name__ == "__main__":
    results = asyncio.run(batch_with_progress())
```

## Batch Scoring

### Text Quality Evaluation in Batches

```python
async def batch_scoring_example():
    """Example of batch text scoring."""
    
    class BatchScorer:
        def __init__(self, base_url: str = "http://localhost:8000"):
            self.base_url = base_url
        
        async def score_batch(
            self,
            texts: List[str],
            metrics: List[str] = None,
            custom_metrics: List[Dict] = None
        ) -> List[Dict[str, Any]]:
            """Score multiple texts for quality."""
            
            if metrics is None:
                metrics = ["sachrichtigkeit", "neutralitaet"]
            
            async with aiohttp.ClientSession() as session:
                tasks = []
                
                for text in texts:
                    data = {
                        "text": text,
                        "predefined_metrics": metrics,
                        "include_improvements": True
                    }
                    
                    if custom_metrics:
                        data["custom_metrics"] = custom_metrics
                    
                    task = session.post(f"{self.base_url}/scoring/evaluate", json=data)
                    tasks.append(task)
                
                responses = await asyncio.gather(*tasks)
                results = []
                
                for response in responses:
                    if response.status == 200:
                        results.append(await response.json())
                    else:
                        error_data = await response.json()
                        results.append({
                            "error": f"HTTP {response.status}",
                            "detail": error_data.get("detail", "Unknown error")
                        })
                    
                    response.close()
                
                return results
    
    # Example texts for scoring
    educational_texts = [
        "Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll.",
        "Machine learning algorithms can be used to predict student performance in online courses.",
        "The French Revolution began in 1789 and had significant impacts on European politics.",
        "DNA contains the genetic instructions for the development of living organisms.",
        "Climate change is affecting global weather patterns and ecosystems worldwide."
    ]
    
    scorer = BatchScorer()
    
    # Custom metric for educational value
    custom_metrics = [
        {
            "name": "educational_value",
            "description": "Assess the educational value and learning potential",
            "criteria": [
                {
                    "name": "clarity",
                    "description": "Information is clearly presented",
                    "weight": 2.0
                },
                {
                    "name": "accuracy",
                    "description": "Information is factually correct",
                    "weight": 2.5
                },
                {
                    "name": "engagement",
                    "description": "Content is engaging for learners",
                    "weight": 1.5
                }
            ]
        }
    ]
    
    results = await scorer.score_batch(
        educational_texts,
        metrics=["sachrichtigkeit"],
        custom_metrics=custom_metrics
    )
    
    # Analyze scoring results
    print("Batch Scoring Results:")
    print("=" * 50)
    
    for i, result in enumerate(results):
        if "error" not in result:
            print(f"\nText {i+1}: {educational_texts[i][:60]}...")
            
            for evaluation in result["results"]:
                metric_name = evaluation["metric_name"]
                score = evaluation["normalized_score"]
                print(f"  {metric_name}: {score:.2%}")
                
                if evaluation.get("suggested_improvements"):
                    print(f"  Improvements: {evaluation['suggested_improvements'][0]}")
        else:
            print(f"\nText {i+1}: Error - {result['error']}")

# Run batch scoring
if __name__ == "__main__":
    asyncio.run(batch_scoring_example())
```

## Data Processing Pipeline

### Complete Educational Content Pipeline

```python
import pandas as pd
from typing import List, Dict, Any
import json

class EducationalContentPipeline:
    """Complete pipeline for processing educational content."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.classifier = BatchClassifier(base_url)
        self.processor = AdvancedBatchProcessor(base_url)
    
    async def process_csv(
        self,
        csv_file: str,
        text_column: str,
        output_file: str = None
    ) -> pd.DataFrame:
        """Process educational content from CSV file."""
        
        # Load data
        df = pd.read_csv(csv_file)
        texts = df[text_column].tolist()
        
        print(f"Processing {len(texts)} texts from {csv_file}")
        
        # Classify texts
        results = await self.processor.process_batch(
            texts,
            mode="skos",
            generate_descriptive_fields=True,
            resource_suggestion=True
        )
        
        # Process results into structured data
        processed_data = []
        
        for i, (text, result) in enumerate(zip(texts, results)):
            if "error" not in result:
                # Extract classifications
                subjects = []
                for vocab_result in result.get("results", []):
                    for classification in vocab_result.get("classifications", []):
                        subjects.append({
                            "subject": classification["preferred_label"],
                            "confidence": classification["confidence"],
                            "vocabulary": vocab_result["vocabulary_name"]
                        })
                
                # Extract metadata
                descriptive = result.get("descriptive_fields", {})
                resource_suggestion = result.get("resource_suggestion_fields", {})
                
                processed_data.append({
                    "original_text": text,
                    "generated_title": descriptive.get("title", ""),
                    "keywords": ", ".join(descriptive.get("keywords", [])),
                    "description": descriptive.get("description", ""),
                    "primary_subject": subjects[0]["subject"] if subjects else "",
                    "subject_confidence": subjects[0]["confidence"] if subjects else 0,
                    "all_subjects": json.dumps(subjects),
                    "learning_phase": resource_suggestion.get("learning_phase", ""),
                    "focus_type": resource_suggestion.get("focus_type", ""),
                    "search_terms": resource_suggestion.get("search_term", ""),
                    "processing_status": "success"
                })
            else:
                processed_data.append({
                    "original_text": text,
                    "processing_status": "error",
                    "error_detail": result.get("detail", result.get("error", "Unknown error"))
                })
        
        # Create output DataFrame
        output_df = pd.DataFrame(processed_data)
        
        if output_file:
            output_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
        return output_df
    
    def analyze_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze processing results."""
        
        successful = df[df["processing_status"] == "success"]
        failed = df[df["processing_status"] == "error"]
        
        analysis = {
            "total_processed": len(df),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(df) * 100,
            "top_subjects": successful["primary_subject"].value_counts().head(10).to_dict(),
            "learning_phases": successful["learning_phase"].value_counts().to_dict(),
            "focus_types": successful["focus_type"].value_counts().to_dict()
        }
        
        return analysis

# Example usage
async def pipeline_example():
    """Example of complete processing pipeline."""
    
    # Create sample data
    sample_data = {
        "content": [
            "Introduction to calculus for engineering students",
            "Photosynthesis and plant biology fundamentals",
            "European history: The Renaissance period",
            "Machine learning algorithms in healthcare",
            "Creative writing: Character development techniques",
            "Statistical analysis for business intelligence",
            "Quantum mechanics: Wave-particle duality",
            "Environmental science: Climate change impacts"
        ],
        "source": ["textbook", "lecture", "article", "research", "workshop", "course", "textbook", "report"]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv("sample_educational_content.csv", index=False)
    
    # Process with pipeline
    pipeline = EducationalContentPipeline()
    
    results_df = await pipeline.process_csv(
        "sample_educational_content.csv",
        "content",
        "processed_educational_content.csv"
    )
    
    # Analyze results
    analysis = pipeline.analyze_results(results_df)
    
    print("\nProcessing Analysis:")
    print(f"Success Rate: {analysis['success_rate']:.1f}%")
    print(f"Top Subjects: {list(analysis['top_subjects'].keys())[:3]}")
    print(f"Learning Phases: {list(analysis['learning_phases'].keys())}")
    
    return results_df

# Run pipeline example
if __name__ == "__main__":
    results = asyncio.run(pipeline_example())
```

## Performance Optimization

### Batch Processing Best Practices

```python
class OptimizedBatchProcessor:
    """Optimized batch processor with performance monitoring."""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_time": 0,
            "average_response_time": 0
        }
    
    async def process_with_monitoring(
        self,
        texts: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Process batch with performance monitoring."""
        
        start_time = time.time()
        
        # Use optimal batch size based on text length
        avg_text_length = sum(len(text) for text in texts) / len(texts)
        
        if avg_text_length < 1000:
            batch_size = 10  # Smaller texts, larger batches
        elif avg_text_length < 5000:
            batch_size = 5   # Medium texts, medium batches
        else:
            batch_size = 2   # Large texts, smaller batches
        
        processor = AdvancedBatchProcessor(
            max_concurrent=batch_size,
            rate_limit_per_minute=60
        )
        
        results = await processor.process_batch(texts, **kwargs)
        
        end_time = time.time()
        
        # Update metrics
        self.metrics["total_requests"] += len(texts)
        self.metrics["successful_requests"] += len([r for r in results if "error" not in r])
        self.metrics["failed_requests"] += len([r for r in results if "error" in r])
        self.metrics["total_time"] += (end_time - start_time)
        self.metrics["average_response_time"] = self.metrics["total_time"] / self.metrics["total_requests"]
        
        return {
            "results": results,
            "processing_time": end_time - start_time,
            "batch_size_used": batch_size,
            "success_rate": self.metrics["successful_requests"] / self.metrics["total_requests"] * 100,
            "average_response_time": self.metrics["average_response_time"]
        }

# Performance monitoring example
async def performance_example():
    processor = OptimizedBatchProcessor()
    
    # Test with different text sizes
    short_texts = ["Short text"] * 20
    medium_texts = ["Medium length text " * 50] * 10
    long_texts = ["Very long educational content " * 200] * 5
    
    for text_set, name in [(short_texts, "short"), (medium_texts, "medium"), (long_texts, "long")]:
        result = await processor.process_with_monitoring(
            text_set,
            mode="skos",
            generate_descriptive_fields=True
        )
        
        print(f"\n{name.title()} texts:")
        print(f"  Processing time: {result['processing_time']:.2f}s")
        print(f"  Batch size used: {result['batch_size_used']}")
        print(f"  Success rate: {result['success_rate']:.1f}%")

if __name__ == "__main__":
    asyncio.run(performance_example())
```

This comprehensive guide covers all aspects of batch processing with the Classification API, from basic concurrent processing to advanced pipelines with monitoring and optimization.
