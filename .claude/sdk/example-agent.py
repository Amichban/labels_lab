#!/usr/bin/env python3
"""
Example of using Claude Code SDK to build custom agents
Based on official Claude Code SDK documentation
"""

import os
import asyncio
from typing import List, Dict, Any
from anthropic import AsyncAnthropic

# Note: In production, use the official SDK when available
# pip install anthropic-claude-code


class ProjectAgent:
    """Custom agent for project-specific tasks using Claude Code SDK patterns"""
    
    def __init__(self, api_key: str = None):
        self.client = AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        self.system_prompt = self._load_system_prompt()
        
    def _load_system_prompt(self) -> str:
        """Load system prompt from CLAUDE.md if it exists"""
        if os.path.exists("CLAUDE.md"):
            with open("CLAUDE.md", "r") as f:
                return f"Project context:\n{f.read()}\n\nYou are a helpful coding assistant."
        return "You are a helpful coding assistant following best practices."
    
    async def implement_feature(self, description: str) -> str:
        """Implement a feature based on description"""
        
        prompt = f"""
        Implement the following feature:
        {description}
        
        Follow the project conventions and patterns.
        Return the implementation as code with explanations.
        """
        
        response = await self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    async def review_code(self, code: str, focus_areas: List[str] = None) -> Dict[str, Any]:
        """Review code for quality and security issues"""
        
        focus = ", ".join(focus_areas) if focus_areas else "security, performance, best practices"
        
        prompt = f"""
        Review the following code focusing on {focus}:
        
        ```
        {code}
        ```
        
        Provide structured feedback with:
        1. Critical issues (security, bugs)
        2. Important improvements (performance, maintainability)
        3. Suggestions (style, documentation)
        """
        
        response = await self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse response into structured format
        return self._parse_review(response.content[0].text)
    
    async def fix_issue(self, error_message: str, code_context: str = None) -> str:
        """Fix an issue based on error message and context"""
        
        prompt = f"""
        Fix the following error:
        {error_message}
        
        {"Code context:\n" + code_context if code_context else ""}
        
        Provide:
        1. Root cause analysis
        2. Fixed code
        3. Explanation of the fix
        """
        
        response = await self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    async def generate_tests(self, code: str, test_framework: str = "pytest") -> str:
        """Generate comprehensive tests for code"""
        
        prompt = f"""
        Generate comprehensive {test_framework} tests for:
        
        ```
        {code}
        ```
        
        Include:
        - Happy path tests
        - Edge cases
        - Error conditions
        - Mocks for external dependencies
        """
        
        response = await self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    async def explain_code(self, code: str, audience: str = "developer") -> str:
        """Explain code for different audiences"""
        
        prompt = f"""
        Explain the following code for a {audience}:
        
        ```
        {code}
        ```
        
        Include:
        - What it does
        - How it works
        - Why it's implemented this way
        - Potential improvements
        """
        
        response = await self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def _parse_review(self, review_text: str) -> Dict[str, Any]:
        """Parse review text into structured format"""
        
        # Simple parsing - in production, use more sophisticated parsing
        return {
            "critical": [],
            "important": [],
            "suggestions": [],
            "raw_feedback": review_text
        }


class ConversationalAgent:
    """Agent that maintains conversation context"""
    
    def __init__(self, api_key: str = None):
        self.client = AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        self.messages = []
        self.system_prompt = "You are Claude Code, a helpful coding assistant."
    
    async def chat(self, user_input: str) -> str:
        """Continue conversation with context"""
        
        self.messages.append({"role": "user", "content": user_input})
        
        response = await self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            system=self.system_prompt,
            messages=self.messages
        )
        
        assistant_message = response.content[0].text
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
    
    def reset(self):
        """Reset conversation context"""
        self.messages = []
    
    def save_context(self, filepath: str):
        """Save conversation context to file"""
        import json
        with open(filepath, "w") as f:
            json.dump(self.messages, f, indent=2)
    
    def load_context(self, filepath: str):
        """Load conversation context from file"""
        import json
        with open(filepath, "r") as f:
            self.messages = json.load(f)


# Example usage
async def main():
    """Example of using the custom agents"""
    
    # Initialize project agent
    agent = ProjectAgent()
    
    # Example 1: Implement a feature
    print("Implementing feature...")
    implementation = await agent.implement_feature(
        "Create a FastAPI endpoint for user registration with email validation"
    )
    print(implementation[:500] + "...")
    
    # Example 2: Review code
    print("\nReviewing code...")
    code_to_review = """
    def get_user(user_id):
        query = f"SELECT * FROM users WHERE id = {user_id}"
        return db.execute(query)
    """
    review = await agent.review_code(
        code_to_review,
        focus_areas=["security", "sql injection"]
    )
    print(review)
    
    # Example 3: Fix an issue
    print("\nFixing issue...")
    fix = await agent.fix_issue(
        error_message="TypeError: unsupported operand type(s) for +: 'int' and 'str'",
        code_context="total = price + tax_rate"
    )
    print(fix[:500] + "...")
    
    # Example 4: Conversational agent
    print("\nConversational agent...")
    chat = ConversationalAgent()
    
    response1 = await chat.chat("How do I implement authentication in FastAPI?")
    print(f"Assistant: {response1[:200]}...")
    
    response2 = await chat.chat("Can you show me an example with JWT?")
    print(f"Assistant: {response2[:200]}...")
    
    # Save conversation
    chat.save_context(".claude/conversation.json")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())