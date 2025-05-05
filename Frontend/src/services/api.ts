import { SuggestionResponse } from '../types/suggestion';

export async function suggestAPI(context: string): Promise<SuggestionResponse> {
  try {
    const response = await fetch('https://slfagrouche-thera-guide-ai.hf.space/suggest', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ context }),
    });

    if (!response.ok) {
      if (response.status === 422) {
        throw new Error('Invalid input. Please check your request and try again.');
      }
      
      if (response.status === 500) {
        throw new Error('Server error. Please try again later.');
      }
      
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error('Unable to connect to the API. Please check your internet connection or try again later.');
    }
    
    if (error instanceof Error) {
      throw error;
    }
    
    throw new Error('An unexpected error occurred. Please try again.');
  }
}