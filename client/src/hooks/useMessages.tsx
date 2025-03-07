
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: Date;
};

type MessagesStore = {
  messages: Message[];
  addMessage: (message: Message) => void;
  clearMessages: () => void;
};

export const useMessages = create<MessagesStore>()(
  persist(
    (set) => ({
      messages: [],
      addMessage: (message) => 
        set((state) => ({ 
          messages: [...state.messages, message] 
        })),
      clearMessages: () => set({ messages: [] }),
    }),
    {
      name: 'chat-messages',
      partialize: (state) => ({
        messages: state.messages.map(msg => ({
          ...msg,
          createdAt: msg.createdAt instanceof Date 
            ? msg.createdAt.toISOString() 
            : msg.createdAt
        }))
      }),
      onRehydrateStorage: () => (state) => {
        if (state) {
          state.messages = state.messages.map(msg => ({
            ...msg,
            createdAt: new Date(msg.createdAt)
          }));
        }
      }
    }
  )
);
