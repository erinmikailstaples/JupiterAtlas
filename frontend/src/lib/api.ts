import axios from 'axios';
import { ChatRequest, ChatResponse } from '../types/chat';

export async function sendMessage(chatRequest: ChatRequest): Promise<ChatResponse> {
  try {
    const response = await axios.post<ChatResponse>(
      'https://improve-chatbot.onrender.com/chat',
      chatRequest,
      {
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error sending message:', error);
    throw error;
  }
}