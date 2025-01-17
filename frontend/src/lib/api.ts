import axios, { AxiosError } from 'axios';
import { Message, ChatRequest, ChatResponse } from '../types/chat';

const api = axios.create({
  baseURL: 'https://jupiteratlas.onrender.com/',
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  },
  timeout: 30000
});

// Add request interceptor for debugging
api.interceptors.request.use(request => {
  console.log('Starting Request:', {
    url: request.url,
    method: request.method,
    data: request.data
  });
  return request;
});

export const sendMessage = async (
  question: string,
  messages: Message[]
): Promise<ChatResponse> => {
  try {
    const request: ChatRequest = {
      question,
      messages,
    };
    
    // First try to check if the server is healthy
    const health = await api.get('/health');
    console.log('Health check response:', health.data);
    
    const response = await api.post<ChatResponse>('/chat', request);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('Axios error details:', {
        message: error.message,
        code: error.code,
        config: error.config,
        response: error.response ? {
          status: error.response.status,
          data: error.response.data,
          headers: error.response.headers,
        } : null,
      });
      throw new Error(`Connection failed: ${error.message}`);
    } else {
      console.error('Unexpected error:', error);
    }
    throw error;
  }
};