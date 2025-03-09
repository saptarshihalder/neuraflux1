import React, { useState } from "react";
import { Button } from "../ui/button";
import { Textarea } from "../ui/textarea";
import { RadioGroup, RadioGroupItem } from "../ui/radio-group";
import { Label } from "../ui/label";

export interface MessageInputProps {
  onSend: (message: string, type: string) => void;
}

export function MessageInput({ onSend }: MessageInputProps) {
  const [message, setMessage] = useState("");
  const [questionType, setQuestionType] = useState("general");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim()) {
      onSend(message, questionType);
      setMessage("");
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col space-y-4">
      <div className="flex flex-col space-y-2">
        <RadioGroup 
          value={questionType} 
          onValueChange={setQuestionType}
          className="flex space-x-4"
        >
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="general" id="general" />
            <Label htmlFor="general">General</Label>
          </div>
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="math" id="math" />
            <Label htmlFor="math">Math</Label>
          </div>
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="grammar" id="grammar" />
            <Label htmlFor="grammar">Grammar</Label>
          </div>
        </RadioGroup>
        <Textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder={
            questionType === "math" 
              ? "Ask a math question..." 
              : questionType === "grammar" 
                ? "Ask a grammar question or check a sentence..." 
                : "Type your message..."
          }
          className="min-h-[100px]"
        />
      </div>
      <Button type="submit" className="self-end" disabled={!message.trim()}>
        Send
      </Button>
    </form>
  );
}
