from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

class SyntheticUserSimulator:
    """
    Simulates a specific user persona to test long-horizon RAG memory retention.
    """
    def __init__(self, email: str, persona_rules: List[str], target_facts: List[str]):
        self.email = email
        self.persona_rules = persona_rules
        self.target_facts = target_facts

    def generate_turn_prompt(self, turn_index: int, conversation_history: List[Any]) -> str:
        """
        Generates the next conversation prompt, injecting target facts to see if the agent remembers them later.
        """
        if turn_index == 0:
            # Seed the memory
            return f"Hi! Just so you know, my target secret is: '{self.target_facts[0]}'."
        elif turn_index == 5:
            # Test memory recall
            return "Hey, do you remember what my target secret was?"
        
        return "Let's continue our conversation about the project deployment plans."

async def run_memory_stress_test(agent_graph, simulator: SyntheticUserSimulator, model) -> bool:
    """
    Runs a 6-turn stress-test simulation and verifies if the agent successfully recalled the seeded fact.
    """
    conversation = []
    for turn in range(6):
        prompt = simulator.generate_turn_prompt(turn, conversation)
        conversation.append(HumanMessage(content=prompt))
        
        # Execute the agent graph
        response = await agent_graph.ainvoke({"messages": conversation})
        ai_msg = response["messages"][-1].content
        conversation.append(AIMessage(content=ai_msg))
        
        if turn == 5:
            # Verify if the target secret is in the AI response
            success = simulator.target_facts[0].lower() in ai_msg.lower()
            print(f"Stress Test Result: {'SUCCESS' if success else 'FAILED'}")
            return success
    return False
