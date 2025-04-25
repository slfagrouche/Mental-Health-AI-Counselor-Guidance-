export interface SuggestionResponse {
  context: string;
  response_type: string;
  crisis_flag: boolean;
  confidence: number;
  rag_suggestion: string;
  rag_risk_level: string;
  direct_suggestion: string;
  direct_risk_level: string;
}