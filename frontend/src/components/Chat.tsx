'use client'
import React, { useState, useRef } from 'react'
import { Message, ChatRequest, ChatResponse } from '../types/chat'
import { sendMessage } from '../lib/api'
import axios from 'axios'

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([{
    role: 'assistant',
    content: "DATABASE ACCESSED. READY FOR QUERIES.",
  }])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const suggestedQuestions = [
    "What is the largest moon of Jupiter?",
    "Can life exist on Europa?"
  ]

  const handleSuggestedQuestion = async (question: string) => {
    setInput(question)
    await handleSubmit()
  }

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    if (!input.trim() || loading) return

    try {
      setLoading(true)
      const userMessage: Message = { role: 'user', content: input }
      setMessages(prev => [...prev, userMessage])
      setInput('')

      const chatRequest: ChatRequest = {
        question: input,
        messages: messages
      }

      const response: ChatResponse = await sendMessage(chatRequest)
      
      const botMessage: Message = {
        role: 'assistant',
        content: response.answer,
        metadata: { context_used: !!response.context }
      }
      
      setMessages(prev => [...prev, botMessage])
    } catch (error) {
      console.error('Error:', error)

      let errorMessage = "🚨ERROR: Connection to Jupiter database failed. Please retry.🚨"
      if (axios.isAxiosError(error) && error.response) {
        errorMessage = `ERROR: ${error.response.data.detail || errorMessage}`
      }

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: errorMessage,
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-black text-green-500 p-4">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold mb-4 text-center text-cyan-400">
          JUPITER ATLAS EXPLORER v1.0 👩‍🚀🪐🔭
        </h1>
        <div className="bg-gray-900/80 border-2 border-cyan-400 rounded-lg p-4 mb-4 h-[600px] overflow-auto">
          {messages.map((message, index) => (
            <div key={index} className={message.role === 'user' ? 'text-yellow-400' : 'text-green-500'}>
              <div className="font-bold">
                {message.role === 'user' ? '> QUERY:' : '> JUPITER.DB:'}
              </div>
              <div className="ml-4">{message.content}</div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
        <div className="flex gap-2 mb-4">
          {suggestedQuestions.map((question, index) => (
            <button
              key={index}
              onClick={() => handleSuggestedQuestion(question)}
              className="bg-cyan-600 text-white px-4 py-2 rounded-lg hover:bg-cyan-700"
            >
              {question}
            </button>
          ))}
        </div>
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="flex-grow bg-gray-800 text-green-500 border-2 border-cyan-400 rounded-lg p-2"
            placeholder="Enter your query..."
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading}
            className="bg-cyan-600 text-white px-6 py-2 rounded-lg hover:bg-cyan-700"
          >
            {loading ? 'CALCULATING...' : 'SUBMIT'}
          </button>
        </form>
      </div>
    </div>
  )
}