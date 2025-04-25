import { useState } from 'react';
import { SuggestionResponse } from '../types/suggestion';
import { suggestAPI } from '../services/api';

export const useSuggestion = () => {
  const [suggestion, setSuggestion] = useState<SuggestionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const submitChallenge = async (challenge: string) => {
    setLoading(true);
    setError('');
    
    try {
      const data = await suggestAPI(challenge);
      setSuggestion(data);
    } catch (err) {
      setError(err instanceof Error 
        ? err.message 
        : 'Failed to fetch suggestion. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const clearError = () => setError('');

  return {
    suggestion,
    loading,
    error,
    submitChallenge,
    clearError
  };
};