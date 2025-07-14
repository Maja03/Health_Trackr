import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../App";

export default function LoginPage() {
  const [isSignup, setIsSignup] = useState(false);
  const [userType, setUserType] = useState("jobseeker");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [message, setMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();
  const { login } = useAuth();

  async function handleSubmit(e) {
    e.preventDefault();
    setIsLoading(true);
    setMessage("");
    
    // Validation
    if (isSignup && password !== confirmPassword) {
      setMessage("Passwords do not match");
      setIsLoading(false);
      return;
    }
    
    if (password.length < 6) {
      setMessage("Password must be at least 6 characters long");
      setIsLoading(false);
      return;
    }
    
    try {
      const endpoint = isSignup ? "/api/signup" : "/api/login";
      const body = isSignup 
        ? new URLSearchParams({ email, password, user_type: userType })
        : new URLSearchParams({ email, password });
        
      const response = await fetch(`http://127.0.0.1:8000${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: body,
      });
      
      const data = await response.json();

      if (response.ok) {
        const successMessage = isSignup ? "Signup successful! Redirecting to dashboard..." : "Login successful! Redirecting to dashboard...";
        setMessage(successMessage);
        
        login({
          email,
          userType: data.user_type || userType,
          token: data.token,
          userId: data.user_id
        });
        
        // Redirect to dashboard after successful login/signup
        setTimeout(() => {
          navigate("/dashboard");
        }, 1000);
      } else {
        setMessage(data.detail || data.error || `${isSignup ? 'Signup' : 'Login'} failed`);
      }
    } catch (error) {
      setMessage("Error: " + error.message);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="max-w-md mx-auto mt-8 bg-white border p-6 rounded shadow">
      <h2 className="text-xl font-bold mb-4 text-center">
        {isSignup ? "Create Account" : "Login"}
      </h2>

      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label className="block mb-1 font-semibold">I am a:</label>
          <select
            className="w-full p-2 border rounded"
            value={userType}
            onChange={(e) => setUserType(e.target.value)}
          >
            <option value="jobseeker">Job Seeker</option>
            <option value="recruiter">Recruiter</option>
          </select>
        </div>

        <input
          type="email"
          placeholder="Email"
          className="w-full mb-3 p-2 border rounded"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password"
          className="w-full mb-3 p-2 border rounded"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        
        {isSignup && (
          <input
            type="password"
            placeholder="Confirm Password"
            className="w-full mb-3 p-2 border rounded"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            required
          />
        )}
        
        <button
          type="submit"
          disabled={isLoading}
          className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-4 py-2 rounded w-full hover:from-blue-700 hover:to-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-all duration-200"
        >
          {isLoading 
            ? (isSignup ? "Creating Account..." : "Logging in...") 
            : (isSignup ? "Create Account" : "Login")
          }
        </button>
      </form>

      <div className="mt-4 text-center">
        <button
          onClick={() => {
            setIsSignup(!isSignup);
            setMessage("");
            setEmail("");
            setPassword("");
            setConfirmPassword("");
          }}
          className="text-blue-600 hover:text-blue-800 underline"
        >
          {isSignup ? "Already have an account? Login" : "Don't have an account? Sign up"}
        </button>
      </div>

      {message && (
        <div className={`mt-4 text-sm text-center font-semibold ${
          message.includes("successful") ? "text-green-600" : "text-red-600"
        }`}>
          {message}
        </div>
      )}
    </div>
  );
}
