import React from 'react';
import { SuggestionResponse } from '../types/suggestion';
import RiskLevelBadge from './RiskLevelBadge';
import { AlertTriangle, CheckCircle, Brain, MessageCircle } from 'lucide-react';

interface ResponseDisplayProps {
  suggestion: SuggestionResponse | null;
  loading: boolean;
}

const ResponseDisplay: React.FC<ResponseDisplayProps> = ({ suggestion, loading }) => {
  if (loading) {
    return (
      <div className="bg-white/80 shadow-lg rounded-xl p-6 animate-pulse border border-gray-100 scale-in backdrop-blur-sm">
        <div className="h-6 bg-gray-200/70 rounded-lg w-3/4 mb-4"></div>
        <div className="space-y-4">
          <div className="h-4 bg-gray-200/70 rounded-lg w-1/2"></div>
          <div className="h-4 bg-gray-200/70 rounded-lg w-5/6"></div>
          <div className="h-4 bg-gray-200/70 rounded-lg w-2/3"></div>
          <div className="h-4 bg-gray-200/70 rounded-lg w-3/4"></div>
          <div className="h-20 bg-gray-200/70 rounded-lg w-full"></div>
        </div>
      </div>
    );
  }

  if (!suggestion) {
    return (
      <div className="bg-white/80 shadow-lg rounded-xl p-8 text-center border border-gray-100 scale-in backdrop-blur-sm">
        <Brain className="w-12 h-12 text-indigo-300 mx-auto mb-4 animate-float" />
        <p className="text-lg text-gray-500">
          Enter a patient challenge above to receive AI-powered suggestions
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white/80 shadow-lg rounded-xl p-6 transition-all border border-gray-100 scale-in backdrop-blur-sm">
      <div className="flex items-center justify-between mb-6 pb-4 border-b border-gray-100 slide-up">
        <h2 className="text-xl font-bold text-gray-800">AI Analysis</h2>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-600">Confidence Score:</span>
          <div className="bg-indigo-100 text-indigo-800 text-sm font-medium px-3 py-1 rounded-full">
            {Math.round(suggestion.confidence * 100)}%
          </div>
        </div>
      </div>

      <div className="space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="bg-gradient-to-br from-white/50 to-indigo-50/50 rounded-xl p-4 hover:shadow-md transition-all duration-200 slide-up backdrop-blur-sm" style={{ animationDelay: '100ms' }}>
            <div className="flex items-center mb-2">
              <MessageCircle className="w-4 h-4 text-indigo-600 mr-2" />
              <span className="text-gray-700 font-medium">Response Type:</span>
              <span className="ml-2 text-gray-800">{suggestion.response_type}</span>
            </div>
          </div>
          
          <div className="bg-gradient-to-br from-white/50 to-indigo-50/50 rounded-xl p-4 hover:shadow-md transition-all duration-200 slide-up backdrop-blur-sm" style={{ animationDelay: '200ms' }}>
            <div className="flex items-center">
              {suggestion.crisis_flag ? (
                <AlertTriangle className="w-4 h-4 text-red-600 mr-2" />
              ) : (
                <CheckCircle className="w-4 h-4 text-green-600 mr-2" />
              )}
              <span className="text-gray-700 font-medium">Crisis Status:</span>
              <span className={`ml-2 font-medium ${suggestion.crisis_flag ? 'text-red-600' : 'text-green-600'}`}>
                {suggestion.crisis_flag ? 'Urgent' : 'Non-Critical'}
              </span>
            </div>
          </div>
        </div>
        
        <div className="border-t border-gray-100 pt-4 slide-up" style={{ animationDelay: '300ms' }}>
          <h3 className="font-medium text-gray-700 mb-3 flex items-center justify-between">
            <span className="flex items-center">
              <Brain className="w-4 h-4 text-indigo-600 mr-2" />
              RAG-Based Suggestion
            </span>
            <RiskLevelBadge level={suggestion.rag_risk_level} />
          </h3>
          <p className="text-gray-800 leading-relaxed bg-gradient-to-br from-white/80 to-indigo-50/80 p-4 rounded-xl border border-indigo-100/50 hover:shadow-md transition-all duration-200 backdrop-blur-sm">
            {suggestion.rag_suggestion}
          </p>
        </div>
        
        <div className="border-t border-gray-100 pt-4 slide-up" style={{ animationDelay: '400ms' }}>
          <h3 className="font-medium text-gray-700 mb-3 flex items-center justify-between">
            <span className="flex items-center">
              <MessageCircle className="w-4 h-4 text-indigo-600 mr-2" />
              Direct Suggestion
            </span>
            <RiskLevelBadge level={suggestion.direct_risk_level} />
          </h3>
          <p className="text-gray-800 leading-relaxed bg-gradient-to-br from-white/80 to-indigo-50/80 p-4 rounded-xl border border-indigo-100/50 hover:shadow-md transition-all duration-200 backdrop-blur-sm">
            {suggestion.direct_suggestion}
          </p>
        </div>
      </div>
    </div>
  );
};

export default ResponseDisplay;