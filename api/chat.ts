
import { insertMessageSchema } from "../shared/schema";
import fetch from "node-fetch";

const PERCHANCE_BASE_URL = "https://perchance.org/api/generators";

async function getPerchanceResponse(prompt: string): Promise<string> {
  try {
    const generators = [
      'writing-prompt-generator',
      'ai-conversation',
      'random-sentence'
    ];

    const responses = await Promise.all(
      generators.map(async (generator) => {
        const response = await fetch(`${PERCHANCE_BASE_URL}/${generator}/generate`);
        if (!response.ok) return null;
        const data = await response.json() as { text?: string };
        return data.text;
      })
    );

    const validResponses = responses.filter(r => r !== null);
    if (validResponses.length > 0) {
      return validResponses[0] || "I apologize, I'm still learning how to respond to that type of question.";
    }

    return "I apologize, I'm having trouble understanding that. Could you try rephrasing your question?";
  } catch (error) {
    console.error("Perchance API error:", error);
    return "I apologize, I'm having trouble understanding that. Could you try rephrasing your question?";
  }
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  try {
    const message = req.body;
    const validatedMessage = insertMessageSchema.parse(message);
    
    // Get response from Perchance
    const perchanceResponse = await getPerchanceResponse(validatedMessage.content);
    
    return res.status(200).json({ 
      type: "response", 
      content: perchanceResponse 
    });
  } catch (error) {
    console.error("API error:", error);
    return res.status(500).json({ 
      type: "error", 
      content: "Something went wrong. Please try again." 
    });
  }
}
