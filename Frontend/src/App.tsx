import React from 'react';
import Header from './components/Header';
import SuggestionForm from './components/SuggestionForm';
import ResponseDisplay from './components/ResponseDisplay';
import { useSuggestion } from './hooks/useSuggestion';

function App() {
  const { 
    suggestion, 
    error, 
    loading, 
    submitChallenge, 
    clearError 
  } = useSuggestion();

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-blue-50 to-sky-50">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,_var(--tw-gradient-stops))] from-blue-100/40 via-transparent to-transparent"></div>
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_bottom_left,_var(--tw-gradient-stops))] from-indigo-100/40 via-transparent to-transparent"></div>
      <Header />
      <main className="container mx-auto px-4 py-8 max-w-3xl relative z-10">
        <SuggestionForm onSubmit={submitChallenge} loading={loading} />
        
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl relative flex items-center backdrop-blur-sm" role="alert">
            <div className="flex-1">
              <strong className="font-bold">Error: </strong>
              <span className="block sm:inline">{error}</span>
            </div>
            <button 
              className="ml-4 text-red-500 hover:text-red-700 transition-colors"
              onClick={clearError}
            >
              <span className="sr-only">Dismiss</span>
              <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}
        
        <ResponseDisplay suggestion={suggestion} loading={loading} />
      </main>
    </div>
  );
}

export default App;