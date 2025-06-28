# SKOS Classification Examples

Examples of using the Classification API with SKOS vocabularies for educational metadata classification.

## Basic SKOS Classification

### Simple Classification

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll.",
    "mode": "skos"
  }'
```

**Response:**
```json
{
  "classification_id": "uuid-string",
  "status": "completed",
  "results": [
    {
      "vocabulary_name": "Discipline",
      "vocabulary_url": "https://w3id.org/kim/hochschulfaechersystematik/scheme",
      "classifications": [
        {
          "concept_uri": "https://w3id.org/kim/hochschulfaechersystematik/n02",
          "preferred_label": "Biology",
          "confidence": 0.95,
          "reasoning": "The text discusses photosynthesis, a fundamental biological process."
        }
      ]
    }
  ]
}
```

## Custom Vocabulary Sources

### Specify Custom SKOS Vocabularies

```python
import requests

data = {
    "text": "Machine learning algorithms for image recognition",
    "mode": "skos",
    "vocabulary_sources": [
        "https://w3id.org/kim/hochschulfaechersystematik/scheme",
        "https://w3id.org/kim/hcrt/scheme"
    ]
}

response = requests.post("http://localhost:8000/classify", json=data)
result = response.json()

for vocab_result in result["results"]:
    print(f"Vocabulary: {vocab_result['vocabulary_name']}")
    for classification in vocab_result["classifications"]:
        print(f"  - {classification['preferred_label']} ({classification['confidence']:.2%})")
```

## Combined Processing

### Classification + Descriptive Fields + Resource Suggestions

```json
{
  "text": "Introduction to quantum mechanics for undergraduate physics students",
  "mode": "skos",
  "generate_descriptive_fields": true,
  "resource_suggestion": true
}
```

**Response:**
```json
{
  "classification_id": "uuid-string",
  "status": "completed",
  "results": [
    {
      "vocabulary_name": "Discipline",
      "classifications": [
        {
          "concept_uri": "https://w3id.org/kim/hochschulfaechersystematik/n17",
          "preferred_label": "Physics",
          "confidence": 0.92
        }
      ]
    }
  ],
  "descriptive_fields": {
    "title": "Introduction to Quantum Mechanics",
    "short_title": "Quantum Mechanics Intro",
    "keywords": ["quantum mechanics", "physics", "undergraduate", "wave-particle duality"],
    "description": "Comprehensive introduction to quantum mechanics principles for undergraduate physics students."
  },
  "resource_suggestion_fields": {
    "focus_type": "content-based",
    "learning_phase": "introduction",
    "filter_suggestions": [
      {
        "vocabulary_url": "https://w3id.org/kim/hochschulfaechersystematik/scheme",
        "suggested_labels": ["Physics", "Theoretical Physics"],
        "confidence": 0.90,
        "reasoning": "Content focuses on fundamental physics concepts"
      }
    ],
    "search_term": "quantum mechanics introduction",
    "keywords": ["quantum physics", "wave function", "uncertainty principle"],
    "title": "Quantum Mechanics Learning Resources",
    "description": "Educational materials for learning quantum mechanics fundamentals"
  }
}
```

## Advanced Examples

### Batch Processing with Python

```python
import requests
import asyncio
import aiohttp
from typing import List, Dict

async def classify_text_async(session: aiohttp.ClientSession, text: str) -> Dict:
    """Classify a single text asynchronously."""
    data = {
        "text": text,
        "mode": "skos",
        "generate_descriptive_fields": True
    }
    
    async with session.post("http://localhost:8000/classify", json=data) as response:
        return await response.json()

async def batch_classify(texts: List[str]) -> List[Dict]:
    """Classify multiple texts concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [classify_text_async(session, text) for text in texts]
        return await asyncio.gather(*tasks)

# Usage
texts = [
    "Photosynthesis in plants",
    "Machine learning algorithms",
    "European history in the 19th century",
    "Organic chemistry reactions"
]

results = asyncio.run(batch_classify(texts))

for i, result in enumerate(results):
    print(f"Text {i+1}: {result['descriptive_fields']['title']}")
    for vocab_result in result["results"]:
        for classification in vocab_result["classifications"]:
            print(f"  - {classification['preferred_label']} ({classification['confidence']:.2%})")
```

### Educational Content Analysis

```python
def analyze_educational_content(content: str) -> Dict:
    """Comprehensive analysis of educational content."""
    
    # Full analysis with all features
    data = {
        "text": content,
        "mode": "skos",
        "vocabulary_sources": [
            "https://w3id.org/kim/hochschulfaechersystematik/scheme",  # Disciplines
            "https://w3id.org/kim/hcrt/scheme",  # Educational contexts
            "https://w3id.org/kim/lrt/scheme"   # Learning resource types
        ],
        "generate_descriptive_fields": True,
        "resource_suggestion": True
    }
    
    response = requests.post("http://localhost:8000/classify", json=data)
    
    if response.status_code == 200:
        result = response.json()
        
        # Extract key information
        analysis = {
            "subject_areas": [],
            "educational_level": [],
            "resource_type": [],
            "metadata": result.get("descriptive_fields", {}),
            "suggestions": result.get("resource_suggestion_fields", {})
        }
        
        # Process classifications by vocabulary
        for vocab_result in result["results"]:
            vocab_name = vocab_result["vocabulary_name"]
            
            for classification in vocab_result["classifications"]:
                if "discipline" in vocab_name.lower():
                    analysis["subject_areas"].append({
                        "subject": classification["preferred_label"],
                        "confidence": classification["confidence"]
                    })
                elif "context" in vocab_name.lower():
                    analysis["educational_level"].append({
                        "level": classification["preferred_label"],
                        "confidence": classification["confidence"]
                    })
                elif "resource" in vocab_name.lower():
                    analysis["resource_type"].append({
                        "type": classification["preferred_label"],
                        "confidence": classification["confidence"]
                    })
        
        return analysis
    else:
        raise Exception(f"API error: {response.status_code}")

# Example usage
content = """
This course introduces students to the fundamental concepts of organic chemistry, 
including molecular structure, bonding, and reaction mechanisms. Students will 
learn about alkanes, alkenes, and aromatic compounds through interactive 
laboratory experiments and problem-solving exercises.
"""

analysis = analyze_educational_content(content)

print("Subject Areas:")
for subject in analysis["subject_areas"]:
    print(f"  - {subject['subject']} ({subject['confidence']:.2%})")

print("\nGenerated Metadata:")
print(f"  Title: {analysis['metadata'].get('title', 'N/A')}")
print(f"  Keywords: {', '.join(analysis['metadata'].get('keywords', []))}")
```

## Error Handling

### Robust Classification with Fallbacks

```python
def classify_with_fallback(text: str, max_retries: int = 3) -> Dict:
    """Classify text with fallback strategies."""
    
    # Primary attempt with full SKOS vocabularies
    primary_data = {
        "text": text,
        "mode": "skos",
        "generate_descriptive_fields": True
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/classify", 
                json=primary_data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 422:
                # Validation error - try with shorter text
                if len(text) > 5000:
                    primary_data["text"] = text[:5000] + "..."
                    continue
                else:
                    raise Exception("Validation error with short text")
            elif response.status_code == 429:
                # Rate limit - wait and retry
                time.sleep(2 ** attempt)
                continue
            else:
                raise Exception(f"API error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            # Timeout - try with minimal processing
            if attempt == max_retries - 1:
                fallback_data = {
                    "text": text[:2000],  # Shorter text
                    "mode": "skos"       # No additional processing
                }
                response = requests.post(
                    "http://localhost:8000/classify",
                    json=fallback_data,
                    timeout=15
                )
                if response.status_code == 200:
                    return response.json()
            else:
                time.sleep(2 ** attempt)
    
    raise Exception("All classification attempts failed")
```

## Performance Optimization

### Vocabulary Caching Strategy

```python
class SKOSClassifier:
    def __init__(self):
        self.default_vocabularies = [
            "https://w3id.org/kim/hochschulfaechersystematik/scheme",
            "https://w3id.org/kim/hcrt/scheme"
        ]
    
    def classify_batch(self, texts: List[str], batch_size: int = 5) -> List[Dict]:
        """Process texts in batches to optimize vocabulary caching."""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch:
                data = {
                    "text": text,
                    "mode": "skos",
                    "vocabulary_sources": self.default_vocabularies
                }
                
                response = requests.post("http://localhost:8000/classify", json=data)
                if response.status_code == 200:
                    batch_results.append(response.json())
                else:
                    batch_results.append({"error": response.status_code})
            
            results.extend(batch_results)
            
            # Small delay between batches to avoid rate limiting
            time.sleep(0.5)
        
        return results
```

## Integration Examples

### Flask Web Application

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_content():
    """Analyze educational content via web interface."""
    
    content = request.json.get('content', '')
    if not content:
        return jsonify({"error": "No content provided"}), 400
    
    try:
        # Classify with SKOS
        classification_data = {
            "text": content,
            "mode": "skos",
            "generate_descriptive_fields": True,
            "resource_suggestion": True
        }
        
        response = requests.post(
            "http://localhost:8000/classify",
            json=classification_data
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": "Classification failed"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

This comprehensive guide covers all aspects of SKOS classification with the API, from basic usage to advanced integration patterns.
