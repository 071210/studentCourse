// server.js
const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs').promises;
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'frontend')));

// API Health Check
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', message: 'API is running', timestamp: new Date().toISOString() });
});

// Recommendation Endpoint
app.post('/api/recommend', async (req, res) => {
  try {
    const input = req.body;
    const result = await runMATLAB(input);
    res.json({ success: true, data: result });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// Run MATLAB and get result
async function runMATLAB(inputData) {
  const inputPath = path.join(__dirname, 'input.json');
  const outputPath = path.join(__dirname, 'output.json');

  await fs.writeFile(inputPath, JSON.stringify(inputData));

  const matlabScript = `
  try
      inputData = jsondecode(fileread('input.json'));
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

      if exist('StudentCourseRecommendationFIS.fis', 'file')
          fis = readfis('StudentCourseRecommendationFIS.fis');
          fisOutput = evalfis(fis, inputArray);
      else
          fisOutput = 3.0;
      end

      courseScores = [0, 0, 0, 0, 0];

      % Subject and Interest scoring
      courseScores(1) = (inputArray(2)/5)*0.7 + (inputArray(3)/5)*1.0 + (inputArray(7)/5);
      courseScores(2) = (inputArray(2)/5)*1.0 + (inputArray(3)/5)*0.8 + (inputArray(6)/5)*0.8 + (inputArray(8)/5);
      courseScores(3) = (inputArray(2)/5)*0.6 + (inputArray(4)/5)*1.0 + (inputArray(9)/5);
      courseScores(4) = (inputArray(2)/5)*0.8 + (inputArray(5)/5)*1.0 + (inputArray(10)/5);
      courseScores(5) = (inputArray(2)/5)*0.9 + (inputArray(6)/5)*1.0 + (inputArray(11)/5);

      % FIS boost
      courseRecommend = round(fisOutput);
      if courseRecommend >= 1 && courseRecommend <= 5
          courseScores(courseRecommend) = courseScores(courseRecommend) + 0.3;
      end

      % Difficulty
      if inputArray(12) == 1
          courseScores([2, 4]) = courseScores([2, 4]) * 1.2;
      elseif inputArray(12) == 2
          courseScores([1, 5]) = courseScores([1, 5]) * 1.1;
      elseif inputArray(12) == 3
          courseScores(3) = courseScores(3) * 1.2;
      end

      % Learning Style
      if inputArray(13) == 1
          courseScores([1, 3, 4]) = courseScores([1, 3, 4]) * 1.1;
      elseif inputArray(13) == 2
          courseScores([1, 2, 4]) = courseScores([1, 2, 4]) * 1.15;
      elseif inputArray(13) == 3
          courseScores([2, 5]) = courseScores([2, 5]) * 1.1;
      elseif inputArray(13) == 4
          courseScores([3, 5]) = courseScores([3, 5]) * 1.05;
      end

      courseScores = max(courseScores, 0.1);
      [sortedScores, sortedIdx] = sort(courseScores, 'descend');

      courseNames = {'Gaming', 'Web Development', 'Fuzzy Logic', 'Database Design', 'Software Validation & Verification'};

      result = struct();
      result.firstRecommendedCourse = courseNames{sortedIdx(1)};
      result.alternativeRecommendedCourse = courseNames{sortedIdx(2)};
      result.firstConfidence = sortedScores(1);
      result.secondConfidence = sortedScores(2);
      result.Confidence_Expert = sortedScores(1);
      result.Confidence_Tree = courseScores(courseRecommend);
      result.probability_Gaming = courseScores(1);
      result.probability_WebDevelopment = courseScores(2);
      result.probability_FuzzyLogic = courseScores(3);
      result.probability_DatabaseDesign = courseScores(4);
      result.probability_SoftwareValidation_Verification = courseScores(5);
      result.expertRecommendation = courseNames{sortedIdx(1)};
      result.treeRecommendation = courseNames{courseRecommend};
      result.finalRecommendation = courseNames{sortedIdx(1)};
      result.fisOutput = fisOutput;
      result.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');

      fid = fopen('output.json', 'w');
      fprintf(fid, '%s', jsonencode(result));
      fclose(fid);
  catch ME
      fid = fopen('output.json', 'w');
      fprintf(fid, '{"error": "%s"}', ME.message);
      fclose(fid);
  end
  exit;
  `;

  await fs.writeFile('script.m', matlabScript);
  await new Promise((resolve, reject) => {
    const matlab = spawn('matlab', ['-batch', "script"]);
    matlab.on('exit', code => (code === 0 ? resolve() : reject(new Error('MATLAB error'))));
  });

  const output = await fs.readFile(outputPath, 'utf8');
  const parsed = JSON.parse(output);
  if (parsed.error) throw new Error(parsed.error);
  return parsed;
}

app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
