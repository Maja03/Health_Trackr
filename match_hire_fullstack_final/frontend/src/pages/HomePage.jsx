import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { FaRocket, FaArrowRight, FaCheckCircle, FaUser } from "react-icons/fa";
import { useAuth } from "../App";

export default function HomePage() {
  const [resume, setResume] = useState("");
  const [job, setJob] = useState("");
  const [resumeFile, setResumeFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const navigate = useNavigate();
  const { isAuthenticated } = useAuth();

  const handleMatch = async () => {
    if (!resume.trim() && !resumeFile) {
      alert("Please provide a resume");
      return;
    }
    if (!job.trim()) {
      alert("Please provide a job description");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("job_description", job);
    if (resumeFile) {
      formData.append("resume_file", resumeFile);
    } else {
      formData.append("resume", resume);
    }

    try {
      const response = await fetch("http://127.0.0.1:8000/api/v2/match/file", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
      alert("Error during matching. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleAdvancedDashboardClick = () => {
    if (isAuthenticated) {
      navigate("/dashboard");
    } else {
      navigate("/login");
    }
  };

  const handleLoginSignupClick = () => {
    if (isAuthenticated) {
      navigate("/dashboard");
    } else {
      navigate("/login");
    }
  };

  const scrollToDemo = () => {
    const demoSection = document.getElementById('quick-demo');
    if (demoSection) {
      demoSection.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      {/* Header with Login Button */}
      <header className="bg-white shadow-sm">
        <div className="max-w-6xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <div className="text-2xl">🤖</div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Advanced MatchHire
            </h1>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="text-center py-20 px-6">
        <div className="max-w-4xl mx-auto">
          <div className="flex justify-center mb-6">
            <div className="text-6xl">🤖</div>
          </div>
          <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-6">
            Advanced MatchHire
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            State-of-the-art AI-powered resume-job matching system with advanced machine learning, 
            comprehensive analysis, and actionable insights for both job seekers and recruiters.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button
              onClick={handleAdvancedDashboardClick}
              className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-4 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-300 flex items-center justify-center space-x-2 text-lg font-semibold"
            >
              <FaRocket />
              <span>{isAuthenticated ? "Go to Advanced Dashboard" : "Login for Advanced Dashboard"}</span>
              <FaArrowRight />
            </button>
            <button 
              onClick={scrollToDemo}
              className="bg-white text-blue-600 px-8 py-4 rounded-lg border-2 border-blue-600 hover:bg-blue-50 transition-all duration-300 text-lg font-semibold"
            >
              Watch Demo
            </button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 px-6 bg-white">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">Advanced ML Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="text-center p-6 rounded-lg bg-gradient-to-br from-blue-50 to-blue-100">
              <div className="text-4xl mb-4 text-blue-600">🧠</div>
              <h3 className="text-xl font-semibold mb-3">Multi-Model Ensemble</h3>
              <p className="text-gray-600">
                Combines BERT, RoBERTa, Random Forest, and Neural Networks for superior accuracy
              </p>
            </div>
            
            <div className="text-center p-6 rounded-lg bg-gradient-to-br from-purple-50 to-purple-100">
              <div className="text-4xl mb-4 text-purple-600">📊</div>
              <h3 className="text-xl font-semibold mb-3">Advanced Analytics</h3>
              <p className="text-gray-600">
                Comprehensive skill analysis, sentiment analysis, and readability metrics
              </p>
            </div>
            
            <div className="text-center p-6 rounded-lg bg-gradient-to-br from-green-50 to-green-100">
              <div className="text-4xl mb-4 text-green-600">🔍</div>
              <h3 className="text-xl font-semibold mb-3">Explainable AI</h3>
              <p className="text-gray-600">
                SHAP and LIME explanations for transparent, interpretable results
              </p>
            </div>
            
            <div className="text-center p-6 rounded-lg bg-gradient-to-br from-orange-50 to-orange-100">
              <div className="text-4xl mb-4 text-orange-600">⚡</div>
              <h3 className="text-xl font-semibold mb-3">Batch Processing</h3>
              <p className="text-gray-600">
                Analyze multiple resumes simultaneously with detailed comparisons
              </p>
            </div>
            
            <div className="text-center p-6 rounded-lg bg-gradient-to-br from-red-50 to-red-100">
              <div className="text-4xl mb-4 text-red-600">📈</div>
              <h3 className="text-xl font-semibold mb-3">Real-time Learning</h3>
              <p className="text-gray-600">
                Continuous model improvement with new data and feedback
              </p>
            </div>
            
            <div className="text-center p-6 rounded-lg bg-gradient-to-br from-indigo-50 to-indigo-100">
              <div className="text-4xl mb-4 text-indigo-600">🎯</div>
              <h3 className="text-xl font-semibold mb-3">Precision Matching</h3>
              <p className="text-gray-600">
                Advanced NLP with named entity recognition and semantic analysis
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Technical Specifications */}
      <section className="py-16 px-6 bg-gray-50">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">Technical Excellence</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            <div>
              <h3 className="text-2xl font-semibold mb-6 text-blue-600">ML Models & Algorithms</h3>
              <div className="space-y-4">
                <div className="flex items-center space-x-3">
                  <FaCheckCircle className="text-green-500" />
                  <span>BERT & RoBERTa Transformers for semantic understanding</span>
                </div>
                <div className="flex items-center space-x-3">
                  <FaCheckCircle className="text-green-500" />
                  <span>Random Forest & Gradient Boosting for ensemble learning</span>
                </div>
                <div className="flex items-center space-x-3">
                  <FaCheckCircle className="text-green-500" />
                  <span>Custom Neural Networks with advanced architectures</span>
                </div>
                <div className="flex items-center space-x-3">
                  <FaCheckCircle className="text-green-500" />
                  <span>Word2Vec & Doc2Vec for document embeddings</span>
                </div>
                <div className="flex items-center space-x-3">
                  <FaCheckCircle className="text-green-500" />
                  <span>LDA Topic Modeling for content analysis</span>
                </div>
                <div className="flex items-center space-x-3">
                  <FaCheckCircle className="text-green-500" />
                  <span>K-Means Clustering for pattern recognition</span>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="text-2xl font-semibold mb-6 text-purple-600">Advanced NLP Pipeline</h3>
              <div className="space-y-4">
                <div className="flex items-center space-x-3">
                  <FaCheckCircle className="text-green-500" />
                  <span>Named Entity Recognition (NER) with spaCy</span>
                </div>
                <div className="flex items-center space-x-3">
                  <FaCheckCircle className="text-green-500" />
                  <span>Advanced skill extraction with KeyBERT & RAKE</span>
                </div>
                <div className="flex items-center space-x-3">
                  <FaCheckCircle className="text-green-500" />
                  <span>Sentiment analysis with VADER & TextBlob</span>
                </div>
                <div className="flex items-center space-x-3">
                  <FaCheckCircle className="text-green-500" />
                  <span>Readability metrics with textstat</span>
                </div>
                <div className="flex items-center space-x-3">
                  <FaCheckCircle className="text-green-500" />
                  <span>Part-of-speech tagging & lemmatization</span>
                </div>
                <div className="flex items-center space-x-3">
                  <FaCheckCircle className="text-green-500" />
                  <span>Multi-language support & preprocessing</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Quick Demo Section */}
      <section id="quick-demo" className="py-16 px-6 bg-white">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-8">Quick Demo</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <textarea
                placeholder="Paste your resume here..."
                className="h-40 w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500"
                value={resume}
                onChange={(e) => {
                  setResume(e.target.value);
                  if (resumeFile) setResumeFile(null);
                }}
                disabled={resumeFile !== null}
              />
              <input
                type="file"
                accept=".pdf,.docx,.txt"
                onChange={(e) => {
                  if (e.target.files.length > 0) {
                    setResumeFile(e.target.files[0]);
                    setResume("");
                  } else {
                    setResumeFile(null);
                  }
                }}
                className="mt-2 w-full"
              />
              {resumeFile && (
                <p className="mt-1 text-sm text-gray-600">Uploaded: {resumeFile.name}</p>
              )}
            </div>
            <textarea
              placeholder="Paste job description here..."
              className="h-40 w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500"
              value={job}
              onChange={(e) => setJob(e.target.value)}
            />
          </div>
          <div className="text-center mt-6">
            <button
              onClick={handleMatch}
              disabled={loading}
              className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 font-semibold"
            >
              {loading ? "Analyzing..." : "Get Basic Match Score"}
            </button>
            <p className="text-sm text-gray-500 mt-2">
              For advanced analysis, visit the Advanced Dashboard
            </p>
          </div>
        </div>
      </section>

      {/* Results Section */}
      {result && (
        <section className="py-8 px-6 bg-green-50">
          <div className="max-w-4xl mx-auto">
            <div className="bg-white rounded-lg p-6 shadow-lg">
              <h3 className="text-2xl font-bold mb-4 text-green-600">
                Basic Match Score: {result.match_score}%
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold text-green-700 mb-2">Matched Keywords:</h4>
                  <div className="flex flex-wrap gap-2">
                    {result.skill_analysis.skill_matches.map((keyword, idx) => (
                      <span key={idx} className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-sm">
                        {keyword}
                      </span>
                    ))}
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold text-red-700 mb-2">Missing Keywords:</h4>
                  <div className="flex flex-wrap gap-2">
                    {result.skill_analysis.missing_skills.map((keyword, idx) => (
                      <span key={idx} className="bg-red-100 text-red-800 px-2 py-1 rounded-full text-sm">
                        {keyword}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
              <div className="mt-4">
                <h4 className="font-semibold mb-2">Suggestions:</h4>
                <ul className="list-disc list-inside space-y-1">
                  {result.detailed_feedback.recommendations.map((s, idx) => (
                    <li key={idx} className="text-gray-700">{s}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* CTA Section */}
      <section className="py-16 px-6 bg-gradient-to-r from-blue-600 to-purple-600 text-white">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6">Ready for Advanced Analysis?</h2>
          <p className="text-xl mb-8 opacity-90">
            Experience the full power of our advanced ML system with comprehensive insights, 
            detailed visualizations, and actionable recommendations.
          </p>
          <Link
            to="/dashboard"
            className="bg-white text-blue-600 px-8 py-4 rounded-lg hover:bg-gray-100 transition-all duration-300 text-lg font-semibold inline-flex items-center space-x-2"
          >
            <FaRocket />
            <span>Launch Advanced Dashboard</span>
            <FaArrowRight />
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-12 px-6">
        <div className="max-w-6xl mx-auto text-center">
          <h3 className="text-2xl font-bold mb-4">Advanced MatchHire</h3>
          <p className="text-gray-300 mb-6">
            State-of-the-art AI-powered resume-job matching with advanced machine learning
          </p>
          <div className="flex justify-center space-x-6 text-sm text-gray-400">
            <span>© 2025 Advanced MatchHire</span>
            <span>•</span>
            <span>Built for smarter career journeys</span>
            <span>•</span>
            <span>Powered by Advanced ML</span>
          </div>
        </div>
      </footer>
    </div>
  );
}