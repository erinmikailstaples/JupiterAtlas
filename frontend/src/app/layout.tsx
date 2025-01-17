import React from 'react'
import './globals.css'
import { Fira_Code } from 'next/font/google'

const firaCode = Fira_Code({ subsets: ['latin'] })

export const metadata = {
  title: 'Jupiter Moons Explorer',
  description: 'A retro-sci-fi chatbot interface for exploring Jupiter\'s moons',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={firaCode.className}>{children}</body>
    </html>
  )
}