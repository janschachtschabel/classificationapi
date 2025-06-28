# Custom Classification Examples

Examples of using the Classification API with custom categories for flexible text classification.

## Basic Custom Classification

### Simple Custom Categories

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This tutorial explains how to build REST APIs with Python and FastAPI framework.",
    "mode": "custom",
    "custom_categories": {
      "Programming Languages": ["Python", "JavaScript", "Java", "C++"],
      "Web Frameworks": ["FastAPI", "Django", "Flask", "Express"],
      "Content Types": ["Tutorial", "Documentation", "Reference", "Example"]
    }
  }'
```

**Response:**
```json
{
  "classification_id": "uuid-string",
  "status": "completed",
  "results": [
    {
      "category_group": "Programming Languages",
      "classifications": [
        {
          "category": "Python",
          "confidence": 0.95,
          "reasoning": "The text explicitly mentions Python as the programming language used with FastAPI."
        }
      ]
    },
    {
      "category_group": "Web Frameworks", 
      "classifications": [
        {
          "category": "FastAPI",
          "confidence": 0.92,
          "reasoning": "FastAPI is specifically mentioned as the framework being taught."
        }
      ]
    },
    {
      "category_group": "Content Types",
      "classifications": [
        {
          "category": "Tutorial",
          "confidence": 0.88,
          "reasoning": "The text describes a tutorial format for learning API development."
        }
      ]
    }
  ]
}
```

## Advanced Custom Classification

### Educational Content Classification

```python
import requests

# Define comprehensive educational categories
educational_categories = {
    "Subject Areas": [
        "Mathematics", "Science", "Computer Science", "Literature", 
        "History", "Geography", "Art", "Music", "Physical Education"
    ],
    "Educational Levels": [
        "Elementary", "Middle School", "High School", "Undergraduate", 
        "Graduate", "Professional Development", "Continuing Education"
    ],
    "Learning Objectives": [
        "Knowledge Acquisition", "Skill Development", "Critical Thinking",
        "Problem Solving", "Creative Expression", "Collaboration"
    ],
    "Content Formats": [
        "Lecture", "Interactive Exercise", "Case Study", "Laboratory",
        "Field Work", "Project-Based", "Assessment", "Review"
    ]
}

def classify_educational_content(text: str) -> dict:
    """Classify educational content using custom categories."""
    
    data = {
        "text": text,
        "mode": "custom",
        "custom_categories": educational_categories,
        "generate_descriptive_fields": True,
        "resource_suggestion": True
    }
    
    response = requests.post("http://localhost:8000/classify", json=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Classification failed: {response.status_code}")

# Example usage
content = """
This interactive chemistry laboratory exercise teaches students about 
acid-base reactions through hands-on experimentation. Students will 
measure pH levels, observe color changes, and calculate reaction rates 
while developing critical thinking skills about chemical processes.
"""

result = classify_educational_content(content)

# Process results
print("Educational Content Analysis:")
print(f"Title: {result.get('descriptive_fields', {}).get('title', 'N/A')}")
print("\nClassifications:")

for category_result in result["results"]:
    category_group = category_result["category_group"]
    print(f"\n{category_group}:")
    
    for classification in category_result["classifications"]:
        print(f"  - {classification['category']} ({classification['confidence']:.2%})")
        print(f"    Reasoning: {classification['reasoning']}")
```

## Specialized Use Cases

### Content Moderation

```python
def moderate_content(text: str) -> dict:
    """Moderate content using custom safety categories."""
    
    moderation_categories = {
        "Content Safety": [
            "Safe for All Ages", "Parental Guidance Suggested", 
            "Adult Content", "Potentially Harmful"
        ],
        "Educational Value": [
            "Highly Educational", "Somewhat Educational", 
            "Entertainment Only", "No Educational Value"
        ],
        "Accuracy Level": [
            "Factually Accurate", "Mostly Accurate", 
            "Contains Inaccuracies", "Misleading Information"
        ],
        "Bias Detection": [
            "Neutral Perspective", "Slight Bias", 
            "Strong Bias", "Propaganda"
        ]
    }
    
    data = {
        "text": text,
        "mode": "custom", 
        "custom_categories": moderation_categories
    }
    
    response = requests.post("http://localhost:8000/classify", json=data)
    return response.json()

# Example
content = "Climate change is a natural phenomenon that has been occurring for millions of years."

moderation_result = moderate_content(content)

for category_result in moderation_result["results"]:
    for classification in category_result["classifications"]:
        if classification["confidence"] > 0.7:
            print(f"⚠️  {category_result['category_group']}: {classification['category']}")
```

### Research Paper Classification

```python
def classify_research_paper(abstract: str) -> dict:
    """Classify research papers by methodology and domain."""
    
    research_categories = {
        "Research Methods": [
            "Experimental", "Observational", "Survey-based", "Case Study",
            "Meta-analysis", "Systematic Review", "Theoretical", "Computational"
        ],
        "Research Domains": [
            "Natural Sciences", "Social Sciences", "Engineering", "Medicine",
            "Computer Science", "Mathematics", "Humanities", "Interdisciplinary"
        ],
        "Study Types": [
            "Quantitative", "Qualitative", "Mixed Methods", "Longitudinal",
            "Cross-sectional", "Comparative", "Exploratory", "Confirmatory"
        ],
        "Impact Areas": [
            "Basic Research", "Applied Research", "Clinical Application",
            "Policy Implications", "Industrial Application", "Educational Impact"
        ]
    }
    
    data = {
        "text": abstract,
        "mode": "custom",
        "custom_categories": research_categories,
        "generate_descriptive_fields": True
    }
    
    response = requests.post("http://localhost:8000/classify", json=data)
    return response.json()

# Example research abstract
abstract = """
This study presents a randomized controlled trial examining the effectiveness 
of machine learning algorithms in predicting student performance in online 
learning environments. We collected data from 1,200 students across multiple 
institutions and applied various classification models to predict academic 
outcomes. Results show significant improvements in prediction accuracy 
compared to traditional statistical methods.
"""

research_result = classify_research_paper(abstract)

print("Research Paper Classification:")
print(f"Generated Title: {research_result['descriptive_fields']['title']}")
print(f"Keywords: {', '.join(research_result['descriptive_fields']['keywords'])}")

for category_result in research_result["results"]:
    print(f"\n{category_result['category_group']}:")
    for classification in category_result["classifications"]:
        print(f"  - {classification['category']} ({classification['confidence']:.2%})")
```

## Batch Processing

### Process Multiple Texts

```python
import asyncio
import aiohttp
from typing import List, Dict

async def classify_batch_custom(texts: List[str], categories: Dict[str, List[str]]) -> List[Dict]:
    """Classify multiple texts with custom categories asynchronously."""
    
    async def classify_single(session: aiohttp.ClientSession, text: str) -> Dict:
        data = {
            "text": text,
            "mode": "custom",
            "custom_categories": categories
        }
        
        async with session.post("http://localhost:8000/classify", json=data) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"error": f"HTTP {response.status}"}
    
    async with aiohttp.ClientSession() as session:
        tasks = [classify_single(session, text) for text in texts]
        return await asyncio.gather(*tasks)

# Example batch processing
content_categories = {
    "Content Types": ["News Article", "Blog Post", "Academic Paper", "Tutorial", "Review"],
    "Topics": ["Technology", "Science", "Business", "Health", "Education"],
    "Sentiment": ["Positive", "Neutral", "Negative", "Mixed"]
}

texts = [
    "New breakthrough in quantum computing promises faster calculations.",
    "Step-by-step guide to building your first web application.",
    "Market analysis shows declining trends in traditional retail.",
    "Recent studies reveal benefits of meditation for mental health."
]

# Run batch classification
results = asyncio.run(classify_batch_custom(texts, content_categories))

for i, result in enumerate(results):
    if "error" not in result:
        print(f"\nText {i+1} Classifications:")
        for category_result in result["results"]:
            best_match = max(category_result["classifications"], key=lambda x: x["confidence"])
            print(f"  {category_result['category_group']}: {best_match['category']} ({best_match['confidence']:.2%})")
```

## Dynamic Category Generation

### Generate Categories from Examples

```python
def generate_categories_from_examples(examples: List[str]) -> Dict[str, List[str]]:
    """Generate custom categories based on example texts."""
    
    # This is a conceptual example - in practice, you might use
    # the classification API itself to analyze examples and suggest categories
    
    categories = {
        "Inferred Topics": [],
        "Content Complexity": ["Beginner", "Intermediate", "Advanced", "Expert"],
        "Format Types": ["Instructional", "Informational", "Analytical", "Narrative"]
    }
    
    # Analyze examples to infer topics (simplified logic)
    common_terms = set()
    for example in examples:
        words = example.lower().split()
        # Extract potential topic keywords (simplified)
        topic_words = [w for w in words if len(w) > 5 and w.isalpha()]
        common_terms.update(topic_words[:3])  # Take first 3 long words
    
    categories["Inferred Topics"] = list(common_terms)[:10]  # Limit to 10
    
    return categories

# Example usage
example_texts = [
    "Machine learning algorithms for data analysis",
    "Python programming tutorial for beginners", 
    "Advanced statistical methods in research",
    "Database design principles and best practices"
]

dynamic_categories = generate_categories_from_examples(example_texts)
print("Generated Categories:", dynamic_categories)

# Use generated categories for classification
new_text = "Introduction to neural networks and deep learning"
result = requests.post("http://localhost:8000/classify", json={
    "text": new_text,
    "mode": "custom",
    "custom_categories": dynamic_categories
})

print("\nClassification with Dynamic Categories:")
for category_result in result.json()["results"]:
    for classification in category_result["classifications"]:
        if classification["confidence"] > 0.5:
            print(f"  {category_result['category_group']}: {classification['category']}")
```

## Error Handling and Validation

### Robust Custom Classification

```python
def robust_custom_classify(text: str, categories: Dict[str, List[str]], max_retries: int = 3) -> Dict:
    """Classify with comprehensive error handling."""
    
    # Validate input
    if not text or len(text.strip()) == 0:
        raise ValueError("Text cannot be empty")
    
    if len(text) > 10000:
        text = text[:10000] + "..."
        print("⚠️  Text truncated to 10,000 characters")
    
    # Validate categories
    for group_name, category_list in categories.items():
        if not category_list or len(category_list) == 0:
            raise ValueError(f"Category group '{group_name}' cannot be empty")
        
        if len(category_list) > 20:
            categories[group_name] = category_list[:20]
            print(f"⚠️  Category group '{group_name}' truncated to 20 items")
    
    # Attempt classification with retries
    for attempt in range(max_retries):
        try:
            data = {
                "text": text,
                "mode": "custom",
                "custom_categories": categories
            }
            
            response = requests.post(
                "http://localhost:8000/classify", 
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 422:
                error_detail = response.json().get("detail", [])
                print(f"Validation error: {error_detail}")
                raise ValueError("Request validation failed")
            elif response.status_code == 429:
                wait_time = 2 ** attempt
                print(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                raise Exception(f"API error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                raise
        except requests.exceptions.ConnectionError:
            print(f"Connection error on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                raise
    
    raise Exception("All classification attempts failed")

# Example with error handling
try:
    result = robust_custom_classify(
        "Sample text for classification",
        {
            "Categories": ["Category1", "Category2", "Category3"],
            "Types": ["Type1", "Type2"]
        }
    )
    print("Classification successful!")
except Exception as e:
    print(f"Classification failed: {e}")
```

This comprehensive guide covers all aspects of custom classification, from basic usage to advanced batch processing and error handling strategies.
