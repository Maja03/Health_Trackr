import React from "react";

export default function AboutPage() {
  return (
    <div>
      <h2 className="text-3xl font-bold mb-4">About Our Team</h2>
      <ul className="list-disc ml-6 space-y-2">
        <li><strong>Member 1</strong>: Frontend Design</li>
        <li><strong>Member 2</strong>: Backend Integration</li>
        <li><strong>Member 3</strong>: AI Matching Logic</li>
        <li><strong>Member 4</strong>: Job Scraper and Resume Parser</li>
        <li><strong>Member 5</strong>: Testing and Deployment</li>
      </ul>
    </div>
  );
}
