import React, { useState } from 'react';
import { Send } from 'lucide-react';

interface SuggestionFormProps {
  onSubmit: (challenge: string) => void;
  loading: boolean;
}

const SuggestionForm: React.FC<SuggestionFormProps> = ({ onSubmit, loading }) => {
  const [challenge, setChallenge] = useState('');
  const [error, setError] = useState('');
  const [isFocused, setIsFocused] = useState(false);

  const handleSubmit = () => {
    if (!challenge.trim()) {
      setError('Please enter a patient challenge');
      return;
    }
    
    setError('');
    onSubmit(challenge);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      handleSubmit();
    }
  };

  return (
    <div className="bg-white/80 shadow-lg rounded-xl p-6 mb-6 transition-all border border-gray-100 scale-in backdrop-blur-sm">
      <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center slide-up">
        Patient Information
        <span className="ml-2 text-sm font-normal text-gray-500">
          (All information is processed securely)
        </span>
      </h2>
      
      <div className="mb-4">
        <label 
          htmlFor="challenge" 
          className="block text-gray-700 font-medium mb-2 slide-up"
          style={{ animationDelay: '100ms' }}
        >
          Describe the patient challenge
        </label>
        <div className="relative">
          <textarea
            id="challenge"
            className={`w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent min-h-32 resize-y pr-24 bg-gradient-to-br from-white/50 to-indigo-50/30 transition-all duration-200 ${
              isFocused ? 'shadow-lg' : 'shadow-sm'
            }`}
            placeholder="Provide details about the patient's situation..."
            value={challenge}
            onChange={(e) => setChallenge(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
          />
          <div className={`absolute bottom-3 right-3 text-xs text-gray-400 bg-white/80 px-2 py-1 rounded-md backdrop-blur-sm transition-opacity duration-200 ${
            isFocused ? 'opacity-100' : 'opacity-0'
          }`}>
            ⌘/Ctrl + Enter to submit
          </div>
        </div>
        {error && (
          <p className="text-red-500 text-sm mt-2 flex items-center slide-up">
            <span className="inline-block w-1 h-1 rounded-full bg-red-500 mr-2"></span>
            {error}
          </p>
        )}
      </div>

      <button
        className="bg-gradient-to-r from-indigo-500 to-blue-600 hover:from-indigo-600 hover:to-blue-700 text-white font-medium py-3 px-6 rounded-xl transition-all duration-300 flex items-center justify-center disabled:opacity-70 disabled:cursor-not-allowed w-full sm:w-auto shadow-md hover:shadow-lg hover:scale-105 active:scale-95"
        onClick={handleSubmit}
        disabled={loading}
      >
        {loading ? (
          <>
            <div className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
            Processing...
          </>
        ) : (
          <>
            <Send className="w-4 h-4 mr-2" />
            Get Suggestion
          </>
        )}
      </button>
    </div>
  );
};

export default SuggestionForm;