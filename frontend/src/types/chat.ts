export interface Message {
    role: 'user' | 'assistant';
    content: string;
    metadata?: {
        context_used?: boolean;
        [key: string]: any;
    };
}

export interface ChatRequest {
    question: string;
    messages: Message[];
}

export interface ChatResponse {
    answer: string;
    context?: string[];
}