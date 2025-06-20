// server.js - Updated with Python MATLAB Logic Integration
const express = require('express');
const cors = require('cors');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs').promises;

const app = express();
const PORT = process.env.PORT || 3000;
const isProduction = process.env.NODE_ENV === 'production';

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('frontend'));

// Logging Middleware
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
  next();
});

// API Health Check Endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString(), message: 'API running with Python MATLAB logic' });
});

// Python MATLAB Logic Integration
class PythonMATLABRecommendationEngine {
  constructor() {
    this.isInitialized = false;
    this.initializeEngine();
  }

  async initializeEngine() {
    try {
      // Check if Python MATLAB logic file exists
      const pythonFile = path.join(__dirname, 'matlab_logic.py');
      
      try {
        await fs.access(pythonFile);
        console.log('✅ Found: matlab_logic.py (Python MATLAB equivalent)');
        this.isInitialized = true;
      } catch (error) {
        console.warn('⚠️ Warning: matlab_logic.py not found. Using JavaScript fallback.');
        this.isInitialized = false;
      }
      
      console.log('🐍 Python MATLAB recommendation engine initialized');
    } catch (error) {
      console.error('Failed to initialize Python MATLAB engine:', error);
      this.isInitialized = false;
    }
  }

  async callPythonMATLABRecommendation(inputData) {
    return new Promise((resolve, reject) => {
      // Create temporary input file
      const tempInputFile = `temp_input_${Date.now()}.json`;
      const tempOutputFile = `temp_output_${Date.now()}.json`;

      // Write input data to temporary file
      fs.writeFile(tempInputFile, JSON.stringify(inputData))
        .then(() => {
          // Create Python script that uses our MATLAB equivalent logic
          const pythonScript = `
import json
import sys
import os

try:
    # Import our MATLAB equivalent engine
    from matlab_logic import CourseRecommendationEngine
    
    # Read input data
    with open('${tempInputFile}', 'r') as f:
        input_data = json.load(f)
    
    print("🐍 Python: Input data loaded successfully")
    print(f"🐍 Python: Input data keys: {list(input_data.keys())}")
    
    # Initialize recommendation engine
    engine = CourseRecommendationEngine()
    print("🐍 Python: Recommendation engine initialized")
    
    # Generate recommendation using exact MATLAB logic
    result = engine.generate_recommendation(input_data)
    print("🐍 Python: Recommendation generated successfully")
    print(f"🐍 Python: Primary recommendation: {result['firstRecommendedCourse']}")
    print(f"🐍 Python: Confidence: {result['firstConfidence']:.3f}")
    print(f"🐍 Python: FIS Output: {result['fisOutput']:.3f}")
    
    # Write result to file
    with open('${tempOutputFile}', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("SUCCESS: Python MATLAB equivalent recommendation generated")
    
except ImportError as e:
    print(f"IMPORT ERROR: {str(e)}")
    print("Make sure matlab_logic.py is in the same directory")
    
    # Write error result
    error_result = {
        "error": f"Import error: {str(e)}",
        "firstRecommendedCourse": "Web Development",
        "alternativeRecommendedCourse": "Database Design", 
        "firstConfidence": 0.5,
        "secondConfidence": 0.3,
        "probability_Gaming": 0.2,
        "probability_WebDevelopment": 0.5,
        "probability_FuzzyLogic": 0.2,
        "probability_DatabaseDesign": 0.3,
        "probability_SoftwareValidation_Verification": 0.2,
        "processingMethod": "Python_Import_Error",
        "fallback": True
    }
    
    with open('${tempOutputFile}', 'w') as f:
        json.dump(error_result, f)

except Exception as e:
    print(f"ERROR: {str(e)}")
    
    # Write error result
    error_result = {
        "error": str(e),
        "firstRecommendedCourse": "Web Development",
        "alternativeRecommendedCourse": "Database Design",
        "firstConfidence": 0.5,
        "secondConfidence": 0.3,
        "probability_Gaming": 0.2,
        "probability_WebDevelopment": 0.5,
        "probability_FuzzyLogic": 0.2,
        "probability_DatabaseDesign": 0.3,
        "probability_SoftwareValidation_Verification": 0.2,
        "processingMethod": "Python_Error",
        "fallback": True
    }
    
    with open('${tempOutputFile}', 'w') as f:
        json.dump(error_result, f)
`;

          // Write Python script to temporary file
          const scriptFile = `temp_script_${Date.now()}.py`;
          return fs.writeFile(scriptFile, pythonScript).then(() => scriptFile);
        })
        .then((scriptFile) => {
          console.log('🐍 Executing Python MATLAB equivalent script...');
          
          // Execute Python script (try python3 first, then python)
          const pythonCommand = isProduction ? 'python3' : 'python3';
          const python = spawn(pythonCommand, [scriptFile], {
            cwd: process.cwd(),
            stdio: ['pipe', 'pipe', 'pipe']
          });

          let output = '';
          let errorOutput = '';

          python.stdout.on('data', (data) => {
            const text = data.toString();
            output += text;
            console.log('🐍 Python:', text.trim());
          });

          python.stderr.on('data', (data) => {
            const text = data.toString();
            errorOutput += text;
            if (!text.includes('UserWarning')) {  // Ignore common warnings
              console.error('⚠️ Python Error:', text.trim());
            }
          });

          python.on('close', async (code) => {
            try {
              // Clean up temporary script file
              await fs.unlink(scriptFile).catch(() => {});
              await fs.unlink(tempInputFile).catch(() => {});

              console.log(`🏁 Python process exited with code: ${code}`);

              if (code === 0) {
                // Read result file
                const resultData = await fs.readFile(tempOutputFile, 'utf8');
                const result = JSON.parse(resultData);
                
                // Clean up output file
                await fs.unlink(tempOutputFile).catch(() => {});
                
                console.log('✅ Python MATLAB equivalent recommendation generated successfully');
                resolve(result);
              } else {
                throw new Error(`Python process exited with code ${code}. Error: ${errorOutput}`);
              }
            } catch (error) {
              // Clean up on error
              await fs.unlink(tempOutputFile).catch(() => {});
              console.error('❌ Python processing failed:', error.message);
              reject(error);
            }
          });

          python.on('error', async (error) => {
            // Clean up on error
            await fs.unlink(scriptFile).catch(() => {});
            await fs.unlink(tempInputFile).catch(() => {});
            await fs.unlink(tempOutputFile).catch(() => {});
            console.error('❌ Python spawn error:', error.message);
            reject(error);
          });
        })
        .catch(reject);
    });
  }

  // Enhanced JavaScript fallback (same as before, but improved)
  generateFallbackRecommendation(inputData) {
    console.log('🔄 Using enhanced JavaScript fallback algorithm');
    const courses = ['Gaming', 'Web Development', 'Fuzzy Logic', 'Database Design', 'Software Validation & Verification'];
    let scores = [0, 0, 0, 0, 0];

    // Enhanced scoring based on your MATLAB logic
    const cgpa = inputData.cgpa || 3;
    const cgpaWeight = cgpa / 5.0;

    // Subject scoring (simplified version of your MATLAB subjectScoring)
    if (inputData.programming >= 1.5) {
      scores[0] += (inputData.programming / 5) * 0.7 * cgpaWeight;  // Gaming
      scores[1] += (inputData.programming / 5) * 1.0 * cgpaWeight;  // Web Development
      scores[2] += (inputData.programming / 5) * 0.6 * cgpaWeight;  // Fuzzy Logic
      scores[3] += (inputData.programming / 5) * 0.8 * cgpaWeight;  // Database
      scores[4] += (inputData.programming / 5) * 0.9 * cgpaWeight;  // Software Testing
    }

    if (inputData.multimedia >= 1.5) {
      scores[0] += (inputData.multimedia / 5) * 1.0 * cgpaWeight;  // Gaming
      scores[1] += (inputData.multimedia / 5) * 0.8 * cgpaWeight;  // Web Development
    }

    if (inputData.machineLearning >= 1.5) {
      scores[2] += (inputData.machineLearning / 5) * 1.0 * cgpaWeight;  // Fuzzy Logic
    }

    if (inputData.database >= 1.5) {
      scores[3] += (inputData.database / 5) * 1.0 * cgpaWeight;  // Database
    }

    if (inputData.softwareEngineering >= 1.5) {
      scores[1] += (inputData.softwareEngineering / 5) * 0.8 * cgpaWeight;  // Web Development
      scores[4] += (inputData.softwareEngineering / 5) * 1.0 * cgpaWeight;  // Software Testing
    }

    // Interest scoring (simplified version of your MATLAB interestScoring)
    if (inputData.gameDevelopment >= 1.5) {
      scores[0] += (inputData.gameDevelopment / 5) * cgpaWeight;
    }
    if (inputData.webDevelopment >= 1.5) {
      scores[1] += (inputData.webDevelopment / 5) * cgpaWeight;
    }
    if (inputData.artificialIntelligence >= 1.5) {
      scores[2] += (inputData.artificialIntelligence / 5) * cgpaWeight;
    }
    if (inputData.databaseSystem >= 1.5) {
      scores[3] += (inputData.databaseSystem / 5) * cgpaWeight;
    }
    if (inputData.softwareValidation >= 1.5) {
      scores[4] += (inputData.softwareValidation / 5) * cgpaWeight;
    }

    // Difficulty and learning style adjustments (exact MATLAB logic)
    const difficulty = inputData.difficulty || 2;
    const learningStyle = inputData.learningStyle || 1;

    if (difficulty === 1) {  // Easy
      scores[1] *= 1.2;  // Web Development
      scores[3] *= 1.2;  // Database Design
    } else if (difficulty === 2) {  // Moderate
      scores[0] *= 1.1;  // Gaming
      scores[4] *= 1.1;  // Software Validation
    } else if (difficulty === 3) {  // Difficult
      scores[2] *= 1.2;  // Fuzzy Logic
    }

    if (learningStyle === 1) {  // Visual
      scores[0] *= 1.1;  // Gaming
      scores[2] *= 1.1;  // Fuzzy Logic
      scores[3] *= 1.1;  // Database Design
    } else if (learningStyle === 2) {  // Kinesthetic
      scores[0] *= 1.15;  // Gaming
      scores[1] *= 1.15;  // Web Development
      scores[3] *= 1.15;  // Database Design
    } else if (learningStyle === 3) {  // Reading/Writing
      scores[1] *= 1.1;  // Web Development
      scores[4] *= 1.1;  // Software Validation
    } else if (learningStyle === 4) {  // Auditory
      scores[2] *= 1.05;  // Fuzzy Logic
      scores[4] *= 1.05;  // Software Validation
    }

    // Ensure minimum scores (exact MATLAB logic)
    scores = scores.map(score => Math.max(score, 0.1));

    // Find top 2 recommendations
    const indexed = scores.map((score, index) => ({ score, index }));
    indexed.sort((a, b) => b.score - a.score);

    return {
      firstRecommendedCourse: courses[indexed[0].index],
      alternativeRecommendedCourse: courses[indexed[1].index],
      firstConfidence: indexed[0].score,
      secondConfidence: indexed[1].score,
      probability_Gaming: scores[0],
      probability_WebDevelopment: scores[1],
      probability_FuzzyLogic: scores[2],
      probability_DatabaseDesign: scores[3],
      probability_SoftwareValidation_Verification: scores[4],
      processingMethod: 'Enhanced_JavaScript_Fallback',
      fallback: true,
      timestamp: new Date().toISOString()
    };
  }
}

// Initialize recommendation engine
const recommendationEngine = new PythonMATLABRecommendationEngine();

// Enhanced Recommendation Endpoint
app.post('/api/recommend', async (req, res) => {
  try {
    const input = req.body;
    console.log('📥 Received input:', input);

    // Validate input data
    const requiredFields = [
      'cgpa', 'programming', 'multimedia', 'machineLearning', 'database',
      'softwareEngineering', 'gameDevelopment', 'webDevelopment',
      'artificialIntelligence', 'databaseSystem', 'softwareValidation',
      'difficulty', 'learningStyle'
    ];

    const missingFields = requiredFields.filter(field => input[field] === undefined || input[field] === null);
    if (missingFields.length > 0) {
      console.error('❌ Missing fields:', missingFields);
      return res.status(400).json({
        success: false,
        error: `Missing required fields: ${missingFields.join(', ')}`
      });
    }

    // Validate data ranges
    if (input.cgpa < 1 || input.cgpa > 5) {
      return res.status(400).json({
        success: false,
        error: 'CGPA must be between 1 and 5'
      });
    }

    let result;

    try {
      // Try Python MATLAB equivalent first
      console.log('🐍 Attempting Python MATLAB equivalent recommendation...');
      result = await recommendationEngine.callPythonMATLABRecommendation(input);
      console.log('✅ Python MATLAB equivalent recommendation successful');
    } catch (error) {
      console.warn('⚠️ Python MATLAB integration failed, using enhanced fallback:', error.message);
      // Use enhanced fallback algorithm
      result = recommendationEngine.generateFallbackRecommendation(input);
    }

    // Add additional metadata
    result.inputData = input;
    result.serverTimestamp = new Date().toISOString();

    console.log('📤 Sending recommendation:', {
      primary: result.firstRecommendedCourse,
      confidence: result.firstConfidence?.toFixed ? result.firstConfidence.toFixed(3) : result.firstConfidence,
      method: result.processingMethod || 'Unknown'
    });

    res.json({
      success: true,
      data: result,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('💥 Recommendation error:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error during recommendation generation',
      details: error.message
    });
  }
});

// Test endpoint for debugging
app.post('/api/test', async (req, res) => {
  const testData = {
    cgpa: 3.5,
    programming: 4,
    multimedia: 2,
    machineLearning: 3,
    database: 4,
    softwareEngineering: 3,
    gameDevelopment: 2,
    webDevelopment: 4,
    artificialIntelligence: 3,
    databaseSystem: 4,
    softwareValidation: 3,
    difficulty: 2,
    learningStyle: 2
  };

  try {
    const result = await recommendationEngine.callPythonMATLABRecommendation(testData);
    res.json({ 
      success: true, 
      data: result,
      note: 'Using Python MATLAB equivalent logic'
    });
  } catch (error) {
    const fallback = recommendationEngine.generateFallbackRecommendation(testData);
    res.json({ 
      success: true, 
      data: fallback, 
      note: 'Used enhanced fallback due to Python error', 
      error: error.message 
    });
  }
});

// Serve HTML pages
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend', 'index.html'));
});

app.get('/input', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend', 'input.html'));
});

app.get('/result', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend', 'result.html'));
});

app.get('/contact', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend', 'contact.html'));
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('💥 Unhandled error:', error);
  res.status(500).json({
    success: false,
    error: 'Internal server error'
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found'
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
  console.log('🐍 Course recommendation system with Python MATLAB equivalent logic');
  console.log('📁 System files:');
  console.log('   ✅ server.js (Node.js backend)');
  console.log('   📊 matlab_logic.py (Python MATLAB equivalent)');
  console.log('   🌐 frontend/ (Web interface)');
  console.log('🔧 Test endpoints:');
  console.log('   - GET  /api/health (health check)');
  console.log('   - POST /api/recommend (main recommendation with Python MATLAB logic)');
  console.log('   - POST /api/test (test with sample data)');
});

module.exports = app;
