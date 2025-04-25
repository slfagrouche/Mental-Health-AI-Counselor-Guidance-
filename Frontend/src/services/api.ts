import { SuggestionResponse } from '../types/suggestion';

export async function suggestAPI(context: string): Promise<SuggestionResponse> {
  try {
    const response = await fetch('/suggest', {
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
      throw new Error('Unable to connect to the server. Please ensure the FastAPI backend server is running at http://127.0.0.1:8000');
    }
    
    if (error instanceof Error) {
      throw error;
    }
    
    throw new Error('An unexpected error occurred. Please try again.');
  }
}