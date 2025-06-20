// server.js
const express = require('express');
const cors = require('cors');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs').promises;

const isProduction = process.env.NODE_ENV === 'production';

const app = express();
const PORT = process.env.PORT || 3000;

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
  res.json({ status: 'OK', timestamp: new Date().toISOString(), message: 'API running' });
});

// MATLAB Integration Functions
class MATLABRecommendationEngine {
  constructor() {
    this.isInitialized = false;
    this.initializeEngine();
  }

  async initializeEngine() {
    try {
      // Check if MATLAB files exist
      const requiredFiles = [
        'StudentCourseRecommendationFIS.fis',
        'CourseDecisionTreeModel.mat'
      ];
      
      for (const file of requiredFiles) {
        try {
          await fs.access(file);
          console.log(`✅ Found: ${file}`);
        } catch (error) {
          console.warn(`⚠️  Warning: ${file} not found. Using fallback algorithm.`);
        }
      }
      
      this.isInitialized = true;
      console.log('Recommendation engine initialized');
    } catch (error) {
      console.error('Failed to initialize recommendation engine:', error);
      this.isInitialized = false;
    }
  }

  async callMATLABRecommendation(inputData) {
    return new Promise((resolve, reject) => {
      // Create temporary input file
      const tempInputFile = `temp_input_${Date.now()}.json`;
      const tempOutputFile = `temp_output_${Date.now()}.json`;

      // Write input data to temporary file
      fs.writeFile(tempInputFile, JSON.stringify(inputData))
        .then(() => {
          // Call MATLAB script
          const matlabScript = `
            try
                % Read input data
                inputData = jsondecode(fileread('${tempInputFile}'));
                
                % Convert to array format expected by your system
                inputArray = [
                    inputData.cgpa,
                    inputData.programming,
                    inputData.multimedia,
                    inputData.machineLearning,
                    inputData.database,
                    inputData.softwareEngineering,
                    inputData.gameDevelopment,
                    inputData.webDevelopment,
                    inputData.artificialIntelligence,
                    inputData.databaseSystem,
                    inputData.softwareValidation,
                    inputData.difficulty,
                    inputData.learningStyle
                ];
                
                % Load FIS and run recommendation
                if exist('StudentCourseRecommendationFIS.fis', 'file')
                    fis = readfis('StudentCourseRecommendationFIS.fis');
                    fisOutput = evalfis(fis, inputArray);
                    fprintf('FIS Output: %.3f\\n', fisOutput);
                else
                    fisOutput = 3.0; % Default
                    fprintf('FIS file not found, using default\\n');
                end
                
                % Load decision tree if available
                if exist('CourseDecisionTreeModel.mat', 'file')
                    try
                        tree = loadLearnerForCoder('CourseDecisionTreeModel');
                        
                        % Ensure inputArray is row vector (1 x 13)
                        if size(inputArray, 1) > 1
                            inputArray = inputArray';
                        end
                        
                        % Ensure fisOutput is scalar
                        if length(fisOutput) > 1
                            fisOutput = fisOutput(1);
                        end
                        
                        % Create tree input as row vector (1 x 14)
                        treeInput = [inputArray(:)', fisOutput];
                        fprintf('Tree input dimensions: %d x %d\\n', size(treeInput));
                        fprintf('Tree input values: [%s]\\n', num2str(treeInput, '%.2f '));
                        
                        [treeIndex, scores] = predict(tree, treeInput);
                        fprintf('Tree prediction: %d\\n', treeIndex);
                        
                        % Ensure scores is the right size
                        if length(scores) ~= 5
                            tempScores = [0.2, 0.2, 0.2, 0.2, 0.2];
                            if treeIndex >= 1 && treeIndex <= 5
                                tempScores(treeIndex) = scores(treeIndex);
                            end
                            scores = tempScores;
                        end
                        
                    catch ME_tree
                        fprintf('Tree prediction error: %s\\n', ME_tree.message);
                        fprintf('Input array size: %d x %d\\n', size(inputArray));
                        fprintf('FIS output size: %d x %d\\n', size(fisOutput));
                        
                        treeIndex = round(fisOutput);
                        treeIndex = min(max(treeIndex, 1), 5);
                        scores = [0.2, 0.2, 0.2, 0.2, 0.2];
                        scores(treeIndex) = 0.8;
                        fprintf('Using FIS output as fallback: %d\\n', treeIndex);
                    end
                else
                    treeIndex = round(fisOutput);
                    treeIndex = min(max(treeIndex, 1), 5);
                    scores = [0.2, 0.2, 0.2, 0.2, 0.2];
                    scores(treeIndex) = 0.8;
                    fprintf('Decision tree not found, using FIS output\\n');
                end
                
                % Run expert system logic (simplified version)
                courseScores = [0, 0, 0, 0, 0];
                
                % Subject scoring
                if inputArray(2) >= 1.5 % Programming
                    courseScores(1) = courseScores(1) + (inputArray(2)/5) * 0.7;
                    courseScores(2) = courseScores(2) + (inputArray(2)/5) * 1.0;
                    courseScores(3) = courseScores(3) + (inputArray(2)/5) * 0.6;
                    courseScores(4) = courseScores(4) + (inputArray(2)/5) * 0.8;
                    courseScores(5) = courseScores(5) + (inputArray(2)/5) * 0.9;
                end
                
                if inputArray(3) >= 1.5 % Multimedia
                    courseScores(1) = courseScores(1) + (inputArray(3)/5) * 1.0;
                    courseScores(2) = courseScores(2) + (inputArray(3)/5) * 0.8;
                end
                
                if inputArray(4) >= 1.5 % Machine Learning
                    courseScores(3) = courseScores(3) + (inputArray(4)/5) * 1.0;
                end
                
                if inputArray(5) >= 1.5 % Database
                    courseScores(4) = courseScores(4) + (inputArray(5)/5) * 1.0;
                end
                
                if inputArray(6) >= 1.5 % Software Engineering
                    courseScores(2) = courseScores(2) + (inputArray(6)/5) * 0.8;
                    courseScores(5) = courseScores(5) + (inputArray(6)/5) * 1.0;
                end
                
                % Interest scoring
                if inputArray(7) >= 1.5 % Game Development
                    courseScores(1) = courseScores(1) + (inputArray(7)/5);
                end
                if inputArray(8) >= 1.5 % Web Development
                    courseScores(2) = courseScores(2) + (inputArray(8)/5);
                end
                if inputArray(9) >= 1.5 % AI
                    courseScores(3) = courseScores(3) + (inputArray(9)/5);
                end
                if inputArray(10) >= 1.5 % Database System
                    courseScores(4) = courseScores(4) + (inputArray(10)/5);
                end
                if inputArray(11) >= 1.5 % Software Validation
                    courseScores(5) = courseScores(5) + (inputArray(11)/5);
                end
                
                % FIS boost
                courseRecommend = round(fisOutput);
                if courseRecommend >= 1 && courseRecommend <= 5 && fisOutput ~= 3.0
                    courseScores(courseRecommend) = courseScores(courseRecommend) + 0.3;
                end
                
                % Adjust by difficulty and learning style
                difficulty = inputArray(12);
                learningStyle = inputArray(13);
                
                switch difficulty
                    case 1
                        courseScores([2, 4]) = courseScores([2, 4]) * 1.2;
                    case 2
                        courseScores([1, 5]) = courseScores([1, 5]) * 1.1;
                    case 3
                        courseScores(3) = courseScores(3) * 1.2;
                end
                
                switch learningStyle
                    case 1
                        courseScores([1, 3, 4]) = courseScores([1, 3, 4]) * 1.1;
                    case 2
                        courseScores([1, 2, 4]) = courseScores([1, 2, 4]) * 1.15;
                    case 3
                        courseScores([2, 5]) = courseScores([2, 5]) * 1.1;
                    case 4
                        courseScores([3, 5]) = courseScores([3, 5]) * 1.05;
                end
                
                courseScores = max(courseScores, 0.1);
                
                % Find top recommendations
                [sortedScores, sortedIdx] = sort(courseScores, 'descend');
                
                % Ensure treeIndex is within bounds
                treeIndex = min(max(round(treeIndex), 1), 5);
                
                % Course names
                courseNames = {'Gaming', 'Web Development', 'Fuzzy Logic', 'Database Design', 'Software Validation & Verification'};
                
                % Create result
                result = struct();
                result.firstRecommendedCourse = courseNames{sortedIdx(1)};
                result.alternativeRecommendedCourse = courseNames{sortedIdx(2)};
                result.firstConfidence = sortedScores(1);
                result.secondConfidence = sortedScores(2);
                result.probability_Gaming = courseScores(1);
                result.probability_WebDevelopment = courseScores(2);
                result.probability_FuzzyLogic = courseScores(3);
                result.probability_DatabaseDesign = courseScores(4);
                result.probability_SoftwareValidation_Verification = courseScores(5);
                result.expertRecommendation = courseNames{sortedIdx(1)};
                result.treeRecommendation = courseNames{treeIndex};
                result.fisOutput = fisOutput;
                
                % Final decision logic
                if sortedIdx(1) == treeIndex
                    result.finalRecommendation = courseNames{sortedIdx(1)};
                else
                    if sortedScores(1) >= 0.75
                        result.finalRecommendation = courseNames{sortedIdx(1)};
                    else
                        result.finalRecommendation = courseNames{treeIndex};
                    end
                end
                
                % Add processing info
                result.processingMethod = 'MATLAB';
                result.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
                
                % Write result to file
                jsonStr = jsonencode(result);
                fid = fopen('${tempOutputFile}', 'w');
                fprintf(fid, '%s', jsonStr);
                fclose(fid);
                
                fprintf('SUCCESS: Recommendation generated\\n');
                
            catch ME
                fprintf('ERROR: %s\\n', ME.message);
                % Write error result
                errorResult = struct();
                errorResult.error = ME.message;
                errorResult.firstRecommendedCourse = 'Web Development';
                errorResult.alternativeRecommendedCourse = 'Database Design';
                errorResult.firstConfidence = 0.5;
                errorResult.secondConfidence = 0.3;
                errorResult.probability_Gaming = 0.2;
                errorResult.probability_WebDevelopment = 0.5;
                errorResult.probability_FuzzyLogic = 0.2;
                errorResult.probability_DatabaseDesign = 0.3;
                errorResult.probability_SoftwareValidation_Verification = 0.2;
                errorResult.processingMethod = 'MATLAB_ERROR';
                
                jsonStr = jsonencode(errorResult);
                fid = fopen('${tempOutputFile}', 'w');
                fprintf(fid, '%s', jsonStr);
                fclose(fid);
            end
            
            exit;
          `;

          // Write MATLAB script to temporary file
          const scriptFile = `temp_script_${Date.now()}.m`;
          return fs.writeFile(scriptFile, matlabScript).then(() => scriptFile);
        })
        .then((scriptFile) => {
          console.log('🔧 Executing MATLAB script...');
          // Execute MATLAB
          const matlab = spawn('matlab', ['-batch', `run('${scriptFile}')`], {
            cwd: process.cwd()
          });

          let output = '';
          let errorOutput = '';

          matlab.stdout.on('data', (data) => {
            const text = data.toString();
            output += text;
            console.log('📊 MATLAB:', text.trim());
          });

          matlab.stderr.on('data', (data) => {
            const text = data.toString();
            errorOutput += text;
            console.error('⚠️ MATLAB Error:', text.trim());
          });

          matlab.on('close', async (code) => {
            try {
              // Clean up temporary script file
              await fs.unlink(scriptFile).catch(() => {});
              await fs.unlink(tempInputFile).catch(() => {});

              console.log(`🏁 MATLAB process exited with code: ${code}`);

              if (code === 0) {
                // Read result file
                const resultData = await fs.readFile(tempOutputFile, 'utf8');
                const result = JSON.parse(resultData);
                
                // Clean up output file
                await fs.unlink(tempOutputFile).catch(() => {});
                
                console.log('✅ MATLAB recommendation generated successfully');
                resolve(result);
              } else {
                throw new Error(`MATLAB process exited with code ${code}: ${errorOutput}`);
              }
            } catch (error) {
              // Clean up on error
              await fs.unlink(tempOutputFile).catch(() => {});
              console.error('❌ MATLAB processing failed:', error.message);
              reject(error);
            }
          });

          matlab.on('error', async (error) => {
            // Clean up on error
            await fs.unlink(scriptFile).catch(() => {});
            await fs.unlink(tempInputFile).catch(() => {});
            await fs.unlink(tempOutputFile).catch(() => {});
            console.error('❌ MATLAB spawn error:', error.message);
            reject(error);
          });
        })
        .catch(reject);
    });
  }

  // Fallback algorithm when MATLAB is not available
  generateFallbackRecommendation(inputData) {
    console.log('🔄 Using JavaScript fallback algorithm');
    const courses = ['Gaming', 'Web Development', 'Fuzzy Logic', 'Database Design', 'Software Validation & Verification'];
    let scores = [0, 0, 0, 0, 0];

    // Simple scoring based on interests and skills
    scores[0] += inputData.gameDevelopment * 0.4 + inputData.programming * 0.3;
    scores[1] += inputData.webDevelopment * 0.4 + inputData.programming * 0.3;
    scores[2] += inputData.artificialIntelligence * 0.4 + inputData.machineLearning * 0.3;
    scores[3] += inputData.databaseSystem * 0.4 + inputData.database * 0.3;
    scores[4] += inputData.softwareValidation * 0.4 + inputData.softwareEngineering * 0.3;

    // CGPA boost
    const cgpaBoost = inputData.cgpa / 5;
    scores = scores.map(score => score + cgpaBoost * 0.2);

    // Difficulty and learning style adjustments
    switch (inputData.difficulty) {
      case 1: // Easy
        scores[1] *= 1.2; // Web Development
        scores[3] *= 1.2; // Database Design
        break;
      case 2: // Moderate
        scores[0] *= 1.1; // Gaming
        scores[4] *= 1.1; // Software Validation
        break;
      case 3: // Difficult
        scores[2] *= 1.2; // Fuzzy Logic
        break;
    }

    switch (inputData.learningStyle) {
      case 1: // Visual
        scores[0] *= 1.1; // Gaming
        scores[2] *= 1.1; // Fuzzy Logic
        scores[3] *= 1.1; // Database Design
        break;
      case 2: // Kinesthetic
        scores[0] *= 1.15; // Gaming
        scores[1] *= 1.15; // Web Development
        scores[3] *= 1.15; // Database Design
        break;
      case 3: // Reading/Writing
        scores[1] *= 1.1; // Web Development
        scores[4] *= 1.1; // Software Validation
        break;
      case 4: // Auditory
        scores[2] *= 1.05; // Fuzzy Logic
        scores[4] *= 1.05; // Software Validation
        break;
    }

    // Normalize scores
    scores = scores.map(score => Math.max(score, 0.1));

    // Find top 2
    const indexed = scores.map((score, index) => ({ score, index }));
    indexed.sort((a, b) => b.score - a.score);

    return {
      firstRecommendedCourse: courses[indexed[0].index],
      alternativeRecommendedCourse: courses[indexed[1].index],
      firstConfidence: Math.min(indexed[0].score, 5.0),
      secondConfidence: Math.min(indexed[1].score, 5.0),
      probability_Gaming: scores[0],
      probability_WebDevelopment: scores[1],
      probability_FuzzyLogic: scores[2],
      probability_DatabaseDesign: scores[3],
      probability_SoftwareValidation_Verification: scores[4],
      processingMethod: 'JavaScript_Fallback',
      fallback: true,
      timestamp: new Date().toISOString()
    };
  }
}

// Initialize recommendation engine
const recommendationEngine = new MATLABRecommendationEngine();

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
      // Try MATLAB integration first
      console.log('🧮 Attempting MATLAB recommendation...');
      result = await recommendationEngine.callMATLABRecommendation(input);
      console.log('✅ MATLAB recommendation successful');
    } catch (error) {
      console.warn('⚠️ MATLAB integration failed, using fallback:', error.message);
      // Use fallback algorithm
      result = recommendationEngine.generateFallbackRecommendation(input);
    }

    // Add additional metadata
    result.inputData = input;
    result.serverTimestamp = new Date().toISOString();

    console.log('📤 Sending recommendation:', {
      primary: result.firstRecommendedCourse,
      confidence: result.firstConfidence,
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
    const result = await recommendationEngine.callMATLABRecommendation(testData);
    res.json({ success: true, data: result });
  } catch (error) {
    const fallback = recommendationEngine.generateFallbackRecommendation(testData);
    res.json({ success: true, data: fallback, note: 'Used fallback due to MATLAB error', error: error.message });
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
  console.log('🤖 Recommendation system initialized with MATLAB integration');
  console.log('📁 Make sure you have these files in your project directory:');
  console.log('   - StudentCourseRecommendationFIS.fis');
  console.log('   - CourseDecisionTreeModel.mat');
  console.log('🔧 Test endpoints:');
  console.log('   - GET  /api/health (health check)');
  console.log('   - POST /api/recommend (main recommendation)');
  console.log('   - POST /api/test (test with sample data)');
});

module.exports = app;