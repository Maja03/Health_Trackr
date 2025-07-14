import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement,
  Filler,
} from 'chart.js';
import { Bar, Doughnut, Line, Radar } from 'react-chartjs-2';
import { 
  FaUpload, 
  FaFileAlt, 
  FaChartBar, 
  FaBrain, 
  FaCogs, 
  FaLightbulb,
  FaCheckCircle,
  FaTimesCircle,
  FaExclamationTriangle,
  FaInfoCircle,
  FaDownload,
  FaSync,
  FaUsers,
  FaSearch,
  FaFilter
} from 'react-icons/fa';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement,
  Filler
);

const AdvancedDashboard = () => {
  const [resumeText, setResumeText] = useState('');
  const [jobDescription, setJobDescription] = useState('');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [matchResult, setMatchResult] = useState(null);
  const [skillAnalysis, setSkillAnalysis] = useState(null);
  const [textAnalysis, setTextAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analysisLevel, setAnalysisLevel] = useState('comprehensive');
  const [batchResumes, setBatchResumes] = useState([]);
  const [batchResults, setBatchResults] = useState(null);
  const [activeTab, setActiveTab] = useState('matching');
  const [modelInfo, setModelInfo] = useState(null);
  const [error, setError] = useState(null);

  const API_BASE = 'http://localhost:8000/api/v2';

  useEffect(() => {
    console.log('AdvancedDashboard component mounted');
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      console.log('Fetching model info...');
      const response = await axios.get(`${API_BASE}/models/info`);
      console.log('Model info received:', response.data);
      setModelInfo(response.data);
    } catch (error) {
      console.error('Error fetching model info:', error);
      setError('Failed to connect to API server. Please make sure the backend is running.');
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    setUploadedFile(file);
  };

  const handleMatch = async () => {
    setLoading(true);
    try {
      let response;
      
      if (uploadedFile) {
        const formData = new FormData();
        formData.append('resume_file', uploadedFile);
        formData.append('job_description', jobDescription);
        formData.append('analysis_level', analysisLevel);
        
        response = await axios.post(`${API_BASE}/match/file`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
      } else {
        response = await axios.post(`${API_BASE}/match`, {
          resume: resumeText,
          job_description: jobDescription,
          analysis_level: analysisLevel
        });
      }
      
      setMatchResult(response.data);
      
      // Fetch additional analyses
      await Promise.all([
        analyzeSkills(resumeText || ''),
        analyzeText(resumeText || '')
      ]);
      
    } catch (error) {
      console.error('Error during matching:', error);
      alert('Error during matching. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const analyzeSkills = async (text) => {
    if (!text.trim()) return;
    
    try {
      const response = await axios.post(`${API_BASE}/analyze-skills`, { text });
      setSkillAnalysis(response.data);
    } catch (error) {
      console.error('Error analyzing skills:', error);
    }
  };

  const analyzeText = async (text) => {
    if (!text.trim()) return;
    
    try {
      const response = await axios.post(`${API_BASE}/analyze-text`, { text });
      setTextAnalysis(response.data);
    } catch (error) {
      console.error('Error analyzing text:', error);
    }
  };

  const handleBatchMatch = async () => {
    if (batchResumes.length === 0) {
      alert('Please add at least one resume to the batch.');
      return;
    }
    
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/batch-match`, {
        resumes: batchResumes,
        job_description: jobDescription,
        analysis_level: analysisLevel
      });
      setBatchResults(response.data);
    } catch (error) {
      console.error('Error during batch matching:', error);
      alert('Error during batch matching. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const addBatchResume = () => {
    if (resumeText.trim()) {
      setBatchResumes([...batchResumes, resumeText]);
      setResumeText('');
    }
  };

  const removeBatchResume = (index) => {
    setBatchResumes(batchResumes.filter((_, i) => i !== index));
  };

  const getScoreColor = (score) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-blue-600';
    if (score >= 0.4) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreIcon = (score) => {
    if (score >= 0.8) return <FaCheckCircle className="text-green-600" />;
    if (score >= 0.6) return <FaInfoCircle className="text-blue-600" />;
    if (score >= 0.4) return <FaExclamationTriangle className="text-yellow-600" />;
    return <FaTimesCircle className="text-red-600" />;
  };

  const renderMatchResult = () => {
    if (!matchResult) return null;

    const modelPredictionsData = {
      labels: Object.keys(matchResult.model_predictions),
      datasets: [{
        label: 'Model Predictions',
        data: Object.values(matchResult.model_predictions),
        backgroundColor: 'rgba(54, 162, 235, 0.8)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1,
      }]
    };

    const skillMatchData = {
      labels: ['Matched Skills', 'Missing Skills'],
      datasets: [{
        data: [
          matchResult.skill_analysis.skill_matches.length,
          matchResult.skill_analysis.missing_skills.length
        ],
        backgroundColor: ['rgba(75, 192, 192, 0.8)', 'rgba(255, 99, 132, 0.8)'],
        borderColor: ['rgba(75, 192, 192, 1)', 'rgba(255, 99, 132, 1)'],
        borderWidth: 1,
      }]
    };

    return (
      <div className="space-y-6">
        {/* Overall Score */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <h3 className="text-xl font-semibold">Match Analysis Results</h3>
            {getScoreIcon(matchResult.match_score)}
          </div>
          
          <div className="mt-4">
            <div className="flex items-center space-x-4">
              <div className="text-4xl font-bold text-blue-600">
                {(matchResult.match_score * 100).toFixed(1)}%
              </div>
              <div>
                <div className="text-lg font-medium">{matchResult.overall_assessment}</div>
                <div className="text-sm text-gray-500">
                  Processed in {matchResult.processing_time.toFixed(2)}s
                </div>
              </div>
            </div>
            
            {/* Progress Bar */}
            <div className="mt-4">
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div 
                  className="bg-blue-600 h-3 rounded-full transition-all duration-500"
                  style={{ width: `${matchResult.match_score * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>

        {/* Model Predictions */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Model Predictions</h3>
          <div className="h-64">
            <Bar 
              data={modelPredictionsData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  y: {
                    beginAtZero: true,
                    max: 1,
                  }
                }
              }}
            />
          </div>
        </div>

        {/* Skill Analysis */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4">Skill Match Distribution</h3>
            <div className="h-64">
              <Doughnut 
                data={skillMatchData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                }}
              />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4">Skill Analysis</h3>
            <div className="space-y-4">
              <div>
                <h4 className="font-medium text-green-600">Matched Skills ({matchResult.skill_analysis.skill_matches.length})</h4>
                <div className="flex flex-wrap gap-2 mt-2">
                  {matchResult.skill_analysis.skill_matches.slice(0, 10).map((skill, index) => (
                    <span key={index} className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-sm">
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
              
              <div>
                <h4 className="font-medium text-red-600">Missing Skills ({matchResult.skill_analysis.missing_skills.length})</h4>
                <div className="flex flex-wrap gap-2 mt-2">
                  {matchResult.skill_analysis.missing_skills.slice(0, 10).map((skill, index) => (
                    <span key={index} className="bg-red-100 text-red-800 px-2 py-1 rounded-full text-sm">
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Detailed Feedback */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Detailed Feedback & Recommendations</h3>
          <div className="space-y-4">
            <div>
              <h4 className="font-medium text-blue-600">Strengths</h4>
              <ul className="list-disc list-inside mt-2 space-y-1">
                {matchResult.detailed_feedback.strengths.slice(0, 5).map((strength, index) => (
                  <li key={index} className="text-gray-700">{strength}</li>
                ))}
              </ul>
            </div>
            
            <div>
              <h4 className="font-medium text-orange-600">Areas for Improvement</h4>
              <ul className="list-disc list-inside mt-2 space-y-1">
                {matchResult.detailed_feedback.recommendations.map((rec, index) => (
                  <li key={index} className="text-gray-700">{rec}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderSkillAnalysis = () => {
    if (!skillAnalysis) return null;

    const skillCategoriesData = {
      labels: ['Technical Skills', 'Soft Skills', 'Programming Languages', 'Frameworks'],
      datasets: [{
        label: 'Skills Found',
        data: [
          skillAnalysis.technical_skills.length,
          skillAnalysis.soft_skills.length,
          skillAnalysis.programming_languages.length,
          skillAnalysis.frameworks.length
        ],
        backgroundColor: [
          'rgba(255, 99, 132, 0.8)',
          'rgba(54, 162, 235, 0.8)',
          'rgba(255, 205, 86, 0.8)',
          'rgba(75, 192, 192, 0.8)'
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 205, 86, 1)',
          'rgba(75, 192, 192, 1)'
        ],
        borderWidth: 1,
      }]
    };

    return (
      <div className="space-y-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Skill Categories Distribution</h3>
          <div className="h-64">
            <Bar 
              data={skillCategoriesData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
              }}
            />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4">Technical Skills</h3>
            <div className="flex flex-wrap gap-2">
              {skillAnalysis.technical_skills.map((skill, index) => (
                <span key={index} className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-sm">
                  {skill}
                </span>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4">Soft Skills</h3>
            <div className="flex flex-wrap gap-2">
              {skillAnalysis.soft_skills.map((skill, index) => (
                <span key={index} className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-sm">
                  {skill}
                </span>
              ))}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4">Programming Languages</h3>
            <div className="flex flex-wrap gap-2">
              {skillAnalysis.programming_languages.map((lang, index) => (
                <span key={index} className="bg-purple-100 text-purple-800 px-2 py-1 rounded-full text-sm">
                  {lang}
                </span>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4">Frameworks & Tools</h3>
            <div className="flex flex-wrap gap-2">
              {skillAnalysis.frameworks.map((framework, index) => (
                <span key={index} className="bg-orange-100 text-orange-800 px-2 py-1 rounded-full text-sm">
                  {framework}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderTextAnalysis = () => {
    if (!textAnalysis) return null;

    // Safely handle missing fields
    const metrics = textAnalysis.readability_metrics ? textAnalysis.readability_metrics : {};
    const readabilityData = {
      labels: Object.keys(metrics),
      datasets: [{
        label: 'Readability Scores',
        data: Object.values(metrics),
        backgroundColor: 'rgba(75, 192, 192, 0.8)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1,
      }]
    };

    const sentiment = textAnalysis.sentiment_analysis ? textAnalysis.sentiment_analysis : {};
    const sentimentData = {
      labels: Object.keys(sentiment),
      datasets: [{
        label: 'Sentiment Scores',
        data: Object.values(sentiment),
        backgroundColor: 'rgba(255, 99, 132, 0.8)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 1,
      }]
    };

    const textStats = textAnalysis.text_statistics ? textAnalysis.text_statistics : {};
    const namedEntities = Array.isArray(textAnalysis.named_entities) ? textAnalysis.named_entities : [];

    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4">Readability Metrics</h3>
            <div className="h-64">
              <Bar 
                data={readabilityData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                }}
              />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4">Sentiment Analysis</h3>
            <div className="h-64">
              <Bar 
                data={sentimentData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    y: {
                      beginAtZero: true,
                      max: 1,
                    }
                  }
                }}
              />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Text Statistics</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(textStats).map(([key, value]) => (
              <div key={key} className="text-center">
                <div className="text-2xl font-bold text-blue-600">{value}</div>
                <div className="text-sm text-gray-600 capitalize">
                  {key.replace(/_/g, ' ')}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Named Entities</h3>
          <div className="flex flex-wrap gap-2">
            {namedEntities.map((entity, index) => (
              <span key={index} className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full text-sm">
                {entity[0]} ({entity[1]})
              </span>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const renderBatchResults = () => {
    if (!batchResults) return null;

    const summary = batchResults && batchResults.summary ? batchResults.summary : {};
    const scoreDist = batchResults.summary && batchResults.summary.score_distribution ? batchResults.summary.score_distribution : {};
    const scoreDistributionData = {
      labels: Object.keys(scoreDist),
      datasets: [{
        label: 'Number of Resumes',
        data: Object.values(scoreDist),
        backgroundColor: [
          'rgba(75, 192, 192, 0.8)',
          'rgba(54, 162, 235, 0.8)',
          'rgba(255, 205, 86, 0.8)',
          'rgba(255, 99, 132, 0.8)'
        ],
        borderColor: [
          'rgba(75, 192, 192, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 205, 86, 1)',
          'rgba(255, 99, 132, 1)'
        ],
        borderWidth: 1,
      }]
    };

    return (
      <div className="space-y-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Batch Analysis Summary</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{summary.total_resumes || 0}</div>
              <div className="text-sm text-gray-600">Total Resumes</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{summary.successful_analyses || 0}</div>
              <div className="text-sm text-gray-600">Successful</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">{(summary.average_score ? (summary.average_score * 100).toFixed(1) : '0.0')}%</div>
              <div className="text-sm text-gray-600">Avg Score</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">{summary.processing_time ? summary.processing_time.toFixed(2) : '0.00'}s</div>
              <div className="text-sm text-gray-600">Processing Time</div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Score Distribution</h3>
          <div className="h-64">
            <Bar 
              data={scoreDistributionData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
              }}
            />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Individual Results</h3>
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {batchResults.results && batchResults.results.map((result, index) => (
              <div key={index} className="border rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <span className="font-medium">Resume {index + 1}</span>
                    {getScoreIcon(result.match_score)}
                  </div>
                  <div className={`text-lg font-bold ${getScoreColor(result.match_score)}`}>
                    {(result.match_score * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="mt-2 text-sm text-gray-600">
                  {result.overall_assessment}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Advanced Resume-Job Matcher</h1>
              <p className="text-gray-600">Sophisticated ML-powered matching with comprehensive analysis</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-500">
                Models: {modelInfo ? [
                  modelInfo.sentence_transformer_loaded,
                  modelInfo.tfidf_vectorizer_loaded,
                  modelInfo.random_forest_loaded
                ].filter(Boolean).length : 0} loaded
              </div>
              <button
                onClick={fetchModelInfo}
                className="p-2 text-gray-400 hover:text-gray-600"
              >
                <FaSync />
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Display */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-md p-4">
            <div className="flex">
              <FaExclamationTriangle className="text-red-400 mt-0.5" />
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">Error</h3>
                <p className="text-sm text-red-700 mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Navigation Tabs */}
        <div className="mb-8">
          <nav className="flex space-x-8">
            {[
              { id: 'matching', name: 'Smart Matching', icon: FaBrain },
              { id: 'batch', name: 'Batch Analysis', icon: FaUsers },
              { id: 'skills', name: 'Skill Analysis', icon: FaCogs },
              { id: 'text', name: 'Text Analysis', icon: FaFileAlt },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium ${
                  activeTab === tab.id
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                <tab.icon />
                <span>{tab.name}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Main Content */}
        {activeTab === 'matching' && (
          <div className="space-y-8">
            {/* Input Section */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4">Resume-Job Matching</h2>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Resume Input */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Resume Content
                  </label>
                  <textarea
                    value={resumeText}
                    onChange={(e) => setResumeText(e.target.value)}
                    placeholder="Paste your resume content here..."
                    className="w-full h-64 p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                  />
                  
                  <div className="mt-2">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Or upload a file
                    </label>
                    <input
                      type="file"
                      accept=".pdf,.docx,.txt"
                      onChange={handleFileUpload}
                      className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                    />
                    {uploadedFile && (
                      <p className="mt-1 text-sm text-gray-600">
                        Selected: {uploadedFile.name}
                      </p>
                    )}
                  </div>
                </div>

                {/* Job Description Input */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Job Description
                  </label>
                  <textarea
                    value={jobDescription}
                    onChange={(e) => setJobDescription(e.target.value)}
                    placeholder="Paste the job description here..."
                    className="w-full h-64 p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              </div>

              {/* Analysis Level */}
              <div className="mt-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Analysis Level
                </label>
                <select
                  value={analysisLevel}
                  onChange={(e) => setAnalysisLevel(e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="basic">Basic Analysis</option>
                  <option value="detailed">Detailed Analysis</option>
                  <option value="comprehensive">Comprehensive Analysis</option>
                </select>
              </div>

              {/* Match Button */}
              <div className="mt-6">
                <button
                  onClick={handleMatch}
                  disabled={loading || (!resumeText.trim() && !uploadedFile) || !jobDescription.trim()}
                  className="w-full bg-blue-600 text-white py-3 px-6 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                      <span>Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <FaSearch />
                      <span>Analyze Match</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Results */}
            {matchResult && renderMatchResult()}
          </div>
        )}

        {activeTab === 'batch' && (
          <div className="space-y-8">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4">Batch Resume Analysis</h2>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Add Resume to Batch
                  </label>
                  <textarea
                    value={resumeText}
                    onChange={(e) => setResumeText(e.target.value)}
                    placeholder="Paste resume content..."
                    className="w-full h-32 p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                  />
                  <button
                    onClick={addBatchResume}
                    disabled={!resumeText.trim()}
                    className="mt-2 bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 disabled:bg-gray-400"
                  >
                    Add to Batch
                  </button>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Job Description
                  </label>
                  <textarea
                    value={jobDescription}
                    onChange={(e) => setJobDescription(e.target.value)}
                    placeholder="Paste the job description..."
                    className="w-full h-32 p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              </div>

              {/* Batch Resumes List */}
              {batchResumes.length > 0 && (
                <div className="mt-6">
                  <h3 className="text-lg font-medium mb-2">Batch Resumes ({batchResumes.length})</h3>
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {batchResumes.map((resume, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-md">
                        <span className="text-sm">Resume {index + 1} ({resume.length} characters)</span>
                        <button
                          onClick={() => removeBatchResume(index)}
                          className="text-red-600 hover:text-red-800"
                        >
                          <FaTimesCircle />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="mt-6">
                <button
                  onClick={handleBatchMatch}
                  disabled={loading || batchResumes.length === 0 || !jobDescription.trim()}
                  className="w-full bg-purple-600 text-white py-3 px-6 rounded-md hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                  {loading ? 'Processing Batch...' : `Analyze ${batchResumes.length} Resumes`}
                </button>
              </div>
            </div>

            {batchResults && renderBatchResults()}
          </div>
        )}

        {activeTab === 'skills' && (
          <div className="space-y-8">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4">Skill Analysis</h2>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Text for Skill Analysis
                </label>
                <textarea
                  value={resumeText}
                  onChange={(e) => setResumeText(e.target.value)}
                  placeholder="Paste text to analyze for skills..."
                  className="w-full h-48 p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                />
                <button
                  onClick={() => analyzeSkills(resumeText)}
                  disabled={!resumeText.trim()}
                  className="mt-4 bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 disabled:bg-gray-400"
                >
                  Analyze Skills
                </button>
              </div>
            </div>

            {skillAnalysis && renderSkillAnalysis()}
          </div>
        )}

        {activeTab === 'text' && (
          <div className="space-y-8">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4">Text Analysis</h2>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Text for Analysis
                </label>
                <textarea
                  value={resumeText}
                  onChange={(e) => setResumeText(e.target.value)}
                  placeholder="Paste text for comprehensive analysis..."
                  className="w-full h-48 p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                />
                <button
                  onClick={() => analyzeText(resumeText)}
                  disabled={!resumeText.trim()}
                  className="mt-4 bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 disabled:bg-gray-400"
                >
                  Analyze Text
                </button>
              </div>
            </div>

            {textAnalysis && renderTextAnalysis()}
          </div>
        )}
      </div>
    </div>
  );
};

export default AdvancedDashboard; 