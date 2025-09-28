# Multimodal RAG API Documentation

## 📋 Overview
This API provides a multimodal Retrieval-Augmented Generation (RAG) system that can process PDFs, images, and audio files, then answer questions based on the uploaded content.

## 🚀 Getting Started

### Base URL
```
http://127.0.0.1:8000
```

### Prerequisites
- Python 3.8+
- Ollama with Qwen2.5VL and Qwen3 models installed
- All dependencies from `requirements.txt`

### Starting the API Server
```bash
uvicorn api.server:app --reload
```

---

## 📚 API Endpoints

### 1. Health Check
**Endpoint:** `GET /`

**Description:** Check if the API server is running

**Response:**
```json
{
  "message": "✅ Multimodal RAG API running"
}
```

**Example:**
```javascript
fetch('http://127.0.0.1:8000/')
  .then(response => response.json())
  .then(data => console.log(data));
```

---

### 2. Upload File
**Endpoint:** `POST /upload/`

**Description:** Upload and process files (PDF, images, audio) into the RAG system

**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (required): File to upload

**Supported File Types:**
- **PDFs:** `.pdf`
- **Images:** `.png`, `.jpg`, `.jpeg` 
- **Audio:** `.mp3`, `.wav`, `.m4a`

**Processing Details:**
- **PDFs:** Text extraction using Qwen2.5VL vision model for OCR
- **Images:** Caption generation and description
- **Audio:** Speech-to-text transcription using Whisper

**Success Response:**
```json
{
  "message": "filename.pdf processed & indexed."
}
```

**Error Response:**
```json
{
  "error": "Unsupported file type"
}
```

**Frontend Examples:**

**React/JavaScript:**
```javascript
const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await fetch('http://127.0.0.1:8000/upload/', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    console.log('Upload result:', result);
    return result;
  } catch (error) {
    console.error('Upload failed:', error);
    throw error;
  }
};

// Usage
const fileInput = document.getElementById('fileInput');
uploadFile(fileInput.files[0]);
```

**HTML Form:**
```html
<form id="uploadForm" enctype="multipart/form-data">
  <input type="file" name="file" accept=".pdf,.png,.jpg,.jpeg,.mp3,.wav,.m4a" required>
  <button type="submit">Upload File</button>
</form>

<script>
document.getElementById('uploadForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  
  const response = await fetch('http://127.0.0.1:8000/upload/', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  console.log(result);
});
</script>
```

**Python:**
```python
import requests

def upload_file(file_path):
    with open(file_path, 'rb') as file:
        files = {'file': file}
        response = requests.post('http://127.0.0.1:8000/upload/', files=files)
        return response.json()

# Usage
result = upload_file('document.pdf')
print(result)
```

---

### 3. Ask Question
**Endpoint:** `POST /ask/`

**Description:** Query the RAG system with natural language questions

**Content-Type:** `application/x-www-form-urlencoded`

**Parameters:**
- `query` (required): Question string

**Response:**
```json
{
  "answer": "AI-generated answer based on uploaded documents"
}
```

---

### 4. Get All Documents
**Endpoint:** `GET /documents/`

**Description:** Retrieve a list of all uploaded documents with metadata

**Response:**
```json
{
  "documents": [
    {
      "filename": "document.pdf",
      "modality": "text",
      "file_size": 1024000,
      "upload_date": "2025-09-27T10:30:45.123456",
      "chunk_count": 15,
      "pages": [1, 2, 3, 4, 5],
      "page_count": 5
    },
    {
      "filename": "image.jpg",
      "modality": "image", 
      "file_size": 512000,
      "upload_date": "2025-09-27T09:15:30.654321",
      "chunk_count": 3,
      "page_count": null
    }
  ],
  "total_count": 2,
  "total_chunks": 18
}
```

**Frontend Example:**
```javascript
const getDocuments = async () => {
  try {
    const response = await fetch('http://127.0.0.1:8000/documents/');
    const data = await response.json();
    console.log('Documents:', data.documents);
    return data;
  } catch (error) {
    console.error('Failed to fetch documents:', error);
    throw error;
  }
};
```

---

### 5. Get Document Details
**Endpoint:** `GET /documents/{filename}`

**Description:** Get detailed information about a specific document including all its chunks

**Parameters:**
- `filename` (path parameter): Name of the document file

**Response:**
```json
{
  "document_info": {
    "filename": "document.pdf",
    "modality": "text",
    "file_size": 1024000,
    "upload_date": "2025-09-27T10:30:45.123456",
    "chunk_count": 15
  },
  "chunks": [
    {
      "chunk_id": "uuid-string-here",
      "page_num": 1,
      "modality": "text",
      "text_excerpt": "This is the beginning of the document content..."
    }
  ]
}
```

**Frontend Example:**
```javascript
const getDocumentDetails = async (filename) => {
  try {
    const response = await fetch(`http://127.0.0.1:8000/documents/${encodeURIComponent(filename)}`);
    const data = await response.json();
    console.log('Document details:', data);
    return data;
  } catch (error) {
    console.error('Failed to fetch document details:', error);
    throw error;
  }
};
```

**Frontend Examples:**

**React/JavaScript:**
```javascript
const askQuestion = async (question) => {
  const formData = new URLSearchParams();
  formData.append('query', question);
  
  try {
    const response = await fetch('http://127.0.0.1:8000/ask/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: formData
    });
    
    const result = await response.json();
    return result.answer;
  } catch (error) {
    console.error('Question failed:', error);
    throw error;
  }
};

// Usage
askQuestion("What is the main topic of the document?")
  .then(answer => console.log('Answer:', answer));
```

**HTML Form:**
```html
<form id="questionForm">
  <input type="text" name="query" placeholder="Ask a question..." required>
  <button type="submit">Ask</button>
</form>
<div id="answer"></div>

<script>
document.getElementById('questionForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new URLSearchParams(new FormData(e.target));
  
  const response = await fetch('http://127.0.0.1:8000/ask/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: formData
  });
  
  const result = await response.json();
  document.getElementById('answer').innerHTML = `<p><strong>Answer:</strong> ${result.answer}</p>`;
});
</script>
```

**jQuery:**
```javascript
function askQuestion(query) {
  $.ajax({
    url: 'http://127.0.0.1:8000/ask/',
    method: 'POST',
    data: { query: query },
    success: function(response) {
      console.log('Answer:', response.answer);
      $('#answer').text(response.answer);
    },
    error: function(xhr, status, error) {
      console.error('Error:', error);
    }
  });
}
```

**Python:**
```python
import requests

def ask_question(query):
    data = {'query': query}
    response = requests.post('http://127.0.0.1:8000/ask/', data=data)
    return response.json()['answer']

# Usage
answer = ask_question("Summarize the uploaded document")
print(f"Answer: {answer}")
```

---

## 🔄 Complete Workflow Example

Here's a complete frontend workflow that uploads a file and then asks questions:

```javascript
class MultimodalRAGClient {
  constructor(baseURL = 'http://127.0.0.1:8000') {
    this.baseURL = baseURL;
  }

  async checkHealth() {
    try {
      const response = await fetch(`${this.baseURL}/`);
      return await response.json();
    } catch (error) {
      throw new Error(`Health check failed: ${error.message}`);
    }
  }

  async uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await fetch(`${this.baseURL}/upload/`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      throw new Error(`Upload error: ${error.message}`);
    }
  }

  async askQuestion(query) {
    const formData = new URLSearchParams();
    formData.append('query', query);
    
    try {
      const response = await fetch(`${this.baseURL}/ask/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`Question failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      return result.answer;
    } catch (error) {
      throw new Error(`Question error: ${error.message}`);
    }
  }
}

// Usage Example
const ragClient = new MultimodalRAGClient();

async function processDocumentAndAsk(file, questions) {
  try {
    // Check if API is running
    await ragClient.checkHealth();
    console.log('✅ API is running');
    
    // Upload file
    console.log('📤 Uploading file...');
    const uploadResult = await ragClient.uploadFile(file);
    console.log('Upload result:', uploadResult);
    
    // Ask questions
    const answers = [];
    for (const question of questions) {
      console.log(`❓ Asking: ${question}`);
      const answer = await ragClient.askQuestion(question);
      console.log(`💡 Answer: ${answer}`);
      answers.push({ question, answer });
    }
    
    return answers;
  } catch (error) {
    console.error('❌ Error:', error.message);
    throw error;
  }
}

// Example usage
const fileInput = document.getElementById('fileInput');
const questions = [
  "What is the main topic of this document?",
  "Summarize the key points",
  "Are there any important dates mentioned?"
];

fileInput.addEventListener('change', async (e) => {
  if (e.target.files[0]) {
    const answers = await processDocumentAndAsk(e.target.files[0], questions);
    displayResults(answers);
  }
});
```

---

## 🎨 React Component Example

```jsx
import React, { useState } from 'react';

const MultimodalRAG = () => {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileUpload = async () => {
    if (!file) return;

    setLoading(true);
    setUploadStatus('Uploading...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://127.0.0.1:8000/upload/', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();
      setUploadStatus(result.message || result.error);
    } catch (error) {
      setUploadStatus('Upload failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAskQuestion = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setAnswer('Processing...');

    const formData = new URLSearchParams();
    formData.append('query', question);

    try {
      const response = await fetch('http://127.0.0.1:8000/ask/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formData
      });

      const result = await response.json();
      setAnswer(result.answer);
    } catch (error) {
      setAnswer('Error: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
      <h1>Multimodal RAG Interface</h1>
      
      {/* File Upload */}
      <div style={{ marginBottom: '20px' }}>
        <h3>1. Upload File</h3>
        <input
          type="file"
          accept=".pdf,.png,.jpg,.jpeg,.mp3,.wav,.m4a"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button 
          onClick={handleFileUpload} 
          disabled={!file || loading}
          style={{ marginLeft: '10px' }}
        >
          {loading ? 'Uploading...' : 'Upload'}
        </button>
        {uploadStatus && <p>{uploadStatus}</p>}
      </div>

      {/* Question Interface */}
      <div>
        <h3>2. Ask Questions</h3>
        <div style={{ display: 'flex', marginBottom: '10px' }}>
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about your uploaded content..."
            style={{ flex: 1, padding: '8px' }}
          />
          <button 
            onClick={handleAskQuestion}
            disabled={loading || !question.trim()}
            style={{ marginLeft: '10px' }}
          >
            {loading ? 'Processing...' : 'Ask'}
          </button>
        </div>
        
        {answer && (
          <div style={{ 
            background: '#f5f5f5', 
            padding: '15px', 
            borderRadius: '5px',
            marginTop: '10px'
          }}>
            <strong>Answer:</strong> {answer}
          </div>
        )}
      </div>
    </div>
  );
};

export default MultimodalRAG;
```

---

## 🛠️ Error Handling

### Common Error Responses:

**File Upload Errors:**
```json
{
  "error": "Unsupported file type"
}
```

**Question Errors:**
- Empty query parameter
- Server processing errors

### Recommended Error Handling:
```javascript
async function safeAPICall(apiFunction) {
  try {
    return await apiFunction();
  } catch (error) {
    console.error('API Error:', error);
    // Show user-friendly error message
    alert('Something went wrong. Please try again.');
    return null;
  }
}
```

---

## 📝 Notes for Frontend Developers

1. **CORS**: The API has CORS enabled for all origins (`allow_origins=["*"]`)

2. **File Size**: Consider implementing file size checks on the frontend before upload

3. **Loading States**: Always show loading indicators for file uploads and questions

4. **File Types**: Validate file types on the frontend to provide immediate feedback

5. **Error Boundaries**: Implement proper error handling for network failures

6. **Progress Indication**: For large files, consider implementing upload progress

---

## 🔧 Testing the API

### Using cURL:
```bash
# Health check
curl http://127.0.0.1:8000/

# Upload file
curl -X POST -F "file=@document.pdf" http://127.0.0.1:8000/upload/

# Ask question
curl -X POST -d "query=What is this document about?" http://127.0.0.1:8000/ask/
```

### Using Postman:
1. **GET** `http://127.0.0.1:8000/` - Health check
2. **POST** `http://127.0.0.1:8000/upload/` - Body: form-data, key: "file", value: select file
3. **POST** `http://127.0.0.1:8000/ask/` - Body: x-www-form-urlencoded, key: "query", value: "your question"

---

## 🚀 Production Considerations

For production deployment, consider:

1. **Authentication**: Add API keys or JWT tokens
2. **Rate Limiting**: Implement request rate limits
3. **File Storage**: Use cloud storage instead of local files
4. **Caching**: Cache frequent queries
5. **Monitoring**: Add logging and monitoring
6. **Validation**: Add stricter input validation
7. **HTTPS**: Use HTTPS in production

---

*This documentation covers all the essentials for frontend integration with the Multimodal RAG API. Happy coding! 🎉*