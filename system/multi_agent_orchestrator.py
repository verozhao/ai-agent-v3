"""
Multi-Agent Orchestrator System
Coordination of multiple intelligent agents with real reasoning and collaboration
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from core.llm_engine import LLMEngine, LLMMessage
from agents.intelligent_agent import IntelligentAgent
from tools.real_tools import registry as tool_registry

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class AgentStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    PLANNING = "planning"
    EXECUTING = "executing"
    COLLABORATING = "collaborating"
    ERROR = "error"

@dataclass
class MultiAgentTask:
    task_id: str
    description: str
    objective: str
    input_data: Dict[str, Any]
    priority: TaskPriority
    required_capabilities: List[str]
    max_agents: int
    timeout_seconds: int
    created_at: datetime
    assigned_agents: List[str] = None
    status: str = "pending"
    results: Dict[str, Any] = None

@dataclass
class AgentCollaboration:
    collaboration_id: str
    participating_agents: List[str]
    collaboration_type: str  # "sequential", "parallel", "consensus"
    shared_context: Dict[str, Any]
    communication_log: List[Dict[str, Any]]
    start_time: datetime
    status: str = "active"

class IntelligentMultiAgentOrchestrator:
    """
    True multi-agent orchestrator that coordinates intelligent agents to solve complex problems
    """
    
    def __init__(self, llm_engine: LLMEngine, max_concurrent_tasks: int = 10):
        self.llm_engine = llm_engine
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Agent management
        self.agents: Dict[str, IntelligentAgent] = {}
        self.agent_status: Dict[str, AgentStatus] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        
        # Task management
        self.active_tasks: Dict[str, MultiAgentTask] = {}
        self.task_queue: List[MultiAgentTask] = []
        self.completed_tasks: List[MultiAgentTask] = []
        
        # Collaboration management
        self.active_collaborations: Dict[str, AgentCollaboration] = {}
        
        # System metrics
        self.metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_completion_time": 0.0,
            "agent_utilization": 0.0,
            "collaboration_success_rate": 0.0,
            "system_uptime": datetime.now()
        }
        
        # System prompt for orchestrator reasoning
        self.system_prompt = self._create_orchestrator_prompt()
        
        logger.info("Initialized Intelligent Multi-Agent Orchestrator")
    
    def _create_orchestrator_prompt(self) -> str:
        """Create system prompt for orchestrator's LLM reasoning"""
        return """You are the Multi-Agent Orchestrator, an advanced AI system that coordinates multiple intelligent agents to solve complex problems.

Your responsibilities:
- Analyze incoming tasks and break them down into subtasks
- Select the optimal agents for each task based on their capabilities
- Design collaboration strategies (sequential, parallel, consensus)
- Monitor agent progress and intervene when needed
- Synthesize results from multiple agents
- Learn from outcomes to improve future orchestration

Available coordination patterns:
- SEQUENTIAL: Agents work in order, each building on previous results
- PARALLEL: Agents work simultaneously on different aspects
- CONSENSUS: Multiple agents solve the same problem for validation
- HIERARCHICAL: Lead agent coordinates sub-agents
- COMPETITIVE: Agents compete to find the best solution

When analyzing tasks, consider:
1. Task complexity and decomposition opportunities
2. Required expertise and agent capabilities
3. Time constraints and dependencies
4. Quality requirements and validation needs
5. Resource utilization and efficiency

Always think strategically about how to achieve the best outcome through intelligent agent coordination."""
    
    def register_agent(self, agent: IntelligentAgent, capabilities: List[str]):
        """Register an intelligent agent with the orchestrator"""
        self.agents[agent.name] = agent
        self.agent_status[agent.name] = AgentStatus.IDLE
        self.agent_capabilities[agent.name] = capabilities
        
        logger.info(f"Registered agent {agent.name} with capabilities: {capabilities}")
    
    async def submit_task(self, 
                         description: str,
                         objective: str,
                         input_data: Dict[str, Any],
                         priority: TaskPriority = TaskPriority.MEDIUM,
                         required_capabilities: List[str] = None,
                         max_agents: int = 3,
                         timeout_seconds: int = 300) -> str:
        """Submit a task to the multi-agent system"""
        
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        task = MultiAgentTask(
            task_id=task_id,
            description=description,
            objective=objective,
            input_data=input_data,
            priority=priority,
            required_capabilities=required_capabilities or [],
            max_agents=max_agents,
            timeout_seconds=timeout_seconds,
            created_at=datetime.now(),
            assigned_agents=[],
            status="pending"
        )
        
        # Add to queue and process
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority.value, reverse=True)
        
        logger.info(f"Submitted task {task_id}: {description}")
        
        # Start processing if capacity available
        if len(self.active_tasks) < self.max_concurrent_tasks:
            asyncio.create_task(self._process_next_task())
        
        return task_id
    
    async def _process_next_task(self):
        """Process the next task in the queue"""
        if not self.task_queue or len(self.active_tasks) >= self.max_concurrent_tasks:
            return
        
        task = self.task_queue.pop(0)
        self.active_tasks[task.task_id] = task
        
        try:
            result = await self._execute_multi_agent_task(task)
            task.results = result
            task.status = "completed"
            
            self.completed_tasks.append(task)
            del self.active_tasks[task.task_id]
            
            # Update metrics
            self.metrics["completed_tasks"] += 1
            self._update_completion_metrics(task)
            
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            task.status = "failed"
            task.results = {"error": str(e)}
            
            self.completed_tasks.append(task)
            del self.active_tasks[task.task_id]
            
            self.metrics["failed_tasks"] += 1
            logger.error(f"Task {task.task_id} failed: {e}")
        
        # Process next task if available
        if self.task_queue:
            asyncio.create_task(self._process_next_task())
    
    async def _execute_multi_agent_task(self, task: MultiAgentTask) -> Dict[str, Any]:
        """Execute a task using multiple coordinated agents"""
        
        logger.info(f"Executing multi-agent task: {task.task_id}")
        
        # Step 1: Analyze task and create coordination strategy
        coordination_strategy = await self._plan_coordination_strategy(task)
        
        # Step 2: Select and assign agents
        assigned_agents = await self._select_agents(task, coordination_strategy)
        task.assigned_agents = [agent.name for agent in assigned_agents]
        
        # Step 3: Execute coordination strategy
        if coordination_strategy["type"] == "sequential":
            result = await self._execute_sequential_coordination(task, assigned_agents, coordination_strategy)
        elif coordination_strategy["type"] == "parallel":
            result = await self._execute_parallel_coordination(task, assigned_agents, coordination_strategy)
        elif coordination_strategy["type"] == "consensus":
            result = await self._execute_consensus_coordination(task, assigned_agents, coordination_strategy)
        else:
            result = await self._execute_default_coordination(task, assigned_agents, coordination_strategy)
        
        # Step 4: Synthesize final results
        final_result = await self._synthesize_multi_agent_results(task, result, coordination_strategy)
        
        return final_result
    
    async def _plan_coordination_strategy(self, task: MultiAgentTask) -> Dict[str, Any]:
        """Use LLM to plan the optimal coordination strategy"""
        
        available_agents = [
            {
                "name": name,
                "capabilities": caps,
                "status": self.agent_status[name].value,
                "recent_performance": agent.metrics
            }
            for name, caps in self.agent_capabilities.items()
            for agent in [self.agents[name]]
        ]
        
        messages = [
            LLMMessage(role="system", content=self.system_prompt),
            LLMMessage(
                role="user",
                content=f"""COORDINATION STRATEGY PLANNING:

Task Details:
- ID: {task.task_id}
- Description: {task.description}
- Objective: {task.objective}
- Input Data: {json.dumps(task.input_data, indent=2)}
- Required Capabilities: {task.required_capabilities}
- Max Agents: {task.max_agents}
- Priority: {task.priority.name}

Available Agents: {json.dumps(available_agents, indent=2)}

Please design the optimal coordination strategy:

1. TASK ANALYSIS:
   - Break down the task into subtasks
   - Identify dependencies and parallelization opportunities
   - Assess complexity and required expertise

2. COORDINATION STRATEGY:
   - Choose coordination type: sequential, parallel, consensus, or hybrid
   - Define agent roles and responsibilities
   - Plan information flow between agents
   - Identify validation and quality control points

3. EXECUTION PLAN:
   - Specify step-by-step execution sequence
   - Define success criteria for each phase
   - Plan error handling and recovery

Respond with a structured coordination strategy in JSON format:
{{
    "type": "sequential|parallel|consensus|hybrid",
    "subtasks": [
        {{
            "id": "subtask_1",
            "description": "what to do",
            "required_capabilities": ["cap1", "cap2"],
            "dependencies": ["subtask_id"],
            "estimated_duration": 30,
            "validation_required": true
        }}
    ],
    "coordination_plan": {{
        "information_flow": "description of how information flows",
        "quality_control": "validation strategy",
        "error_handling": "recovery approach"
    }},
    "success_criteria": "how to measure success",
    "estimated_total_duration": 120
}}"""
            )
        ]
        
        response = await self.llm_engine.complete(
            messages=messages,
            temperature=0.3,
            max_tokens=2000
        )
        
        # Parse strategy from response
        try:
            strategy_text = response.content
            if "```json" in strategy_text:
                json_start = strategy_text.find("```json") + 7
                json_end = strategy_text.find("```", json_start)
                strategy_text = strategy_text[json_start:json_end]
            
            strategy = json.loads(strategy_text)
            
        except json.JSONDecodeError:
            # Fallback strategy
            strategy = {
                "type": "sequential",
                "subtasks": [
                    {
                        "id": "analysis",
                        "description": f"Analyze {task.description}",
                        "required_capabilities": task.required_capabilities,
                        "dependencies": [],
                        "estimated_duration": 60,
                        "validation_required": True
                    }
                ],
                "coordination_plan": {
                    "information_flow": "Sequential processing with validation",
                    "quality_control": "Each step validated before proceeding",
                    "error_handling": "Retry with different agent on failure"
                },
                "success_criteria": "Task objective achieved with high confidence",
                "estimated_total_duration": 120
            }
        
        logger.info(f"Created coordination strategy: {strategy['type']} with {len(strategy['subtasks'])} subtasks")
        
        return strategy
    
    async def _select_agents(self, task: MultiAgentTask, strategy: Dict[str, Any]) -> List[IntelligentAgent]:
        """Intelligently select agents based on task requirements and strategy"""
        
        # Analyze agent suitability using LLM
        agent_analysis = await self._analyze_agent_suitability(task, strategy)
        
        selected_agents = []
        used_agents = set()
        
        for subtask in strategy.get("subtasks", []):
            required_caps = subtask.get("required_capabilities", [])
            
            # Find best available agent for this subtask
            best_agent = None
            best_score = -1
            
            for agent_name, agent_caps in self.agent_capabilities.items():
                if (agent_name not in used_agents and 
                    self.agent_status[agent_name] == AgentStatus.IDLE and
                    any(cap in agent_caps for cap in required_caps)):
                    
                    # Calculate suitability score
                    capability_match = len(set(required_caps) & set(agent_caps)) / max(len(required_caps), 1)
                    performance_score = self.agents[agent_name].metrics.get("successful_tasks", 0) / max(self.agents[agent_name].metrics.get("tasks_completed", 1), 1)
                    
                    total_score = capability_match * 0.7 + performance_score * 0.3
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_agent = self.agents[agent_name]
            
            if best_agent and best_agent not in selected_agents:
                selected_agents.append(best_agent)
                used_agents.add(best_agent.name)
                self.agent_status[best_agent.name] = AgentStatus.BUSY
        
        # Ensure we have at least one agent
        if not selected_agents:
            # Select any available agent
            for agent_name, status in self.agent_status.items():
                if status == AgentStatus.IDLE:
                    selected_agents.append(self.agents[agent_name])
                    self.agent_status[agent_name] = AgentStatus.BUSY
                    break
        
        logger.info(f"Selected {len(selected_agents)} agents: {[a.name for a in selected_agents]}")
        
        return selected_agents
    
    async def _analyze_agent_suitability(self, task: MultiAgentTask, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to analyze agent suitability for the task"""
        
        messages = [
            LLMMessage(role="system", content=self.system_prompt),
            LLMMessage(
                role="user",
                content=f"""AGENT SUITABILITY ANALYSIS:

Task: {task.description}
Strategy: {strategy['type']}
Subtasks: {json.dumps(strategy.get('subtasks', []), indent=2)}

Available Agents:
{json.dumps([
    {
        "name": name,
        "capabilities": caps,
        "performance": self.agents[name].metrics,
        "status": self.agent_status[name].value
    }
    for name, caps in self.agent_capabilities.items()
], indent=2)}

Analyze which agents would be best suited for each subtask and provide recommendations for optimal assignment."""
            )
        ]
        
        response = await self.llm_engine.complete(
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        return {"analysis": response.content}
    
    async def _execute_sequential_coordination(self, task: MultiAgentTask, agents: List[IntelligentAgent], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sequential coordination where agents work in order"""
        
        logger.info(f"Executing sequential coordination with {len(agents)} agents")
        
        results = {}
        accumulated_context = task.input_data.copy()
        
        for i, (agent, subtask) in enumerate(zip(agents, strategy.get("subtasks", []))):
            logger.info(f"Agent {agent.name} executing subtask {i+1}: {subtask.get('description', '')}")
            
            # Prepare task for agent
            agent_task = {
                "type": "subtask_execution",
                "subtask_id": subtask.get("id", f"subtask_{i+1}"),
                "description": subtask.get("description", ""),
                "input_data": accumulated_context,
                "context": results,
                "objective": task.objective,
                "coordination_type": "sequential",
                "position_in_sequence": i + 1,
                "total_agents": len(agents)
            }
            
            # Execute
            agent_result = await agent.process_task(agent_task)
            
            # Store result and update context
            results[f"agent_{i+1}_{agent.name}"] = agent_result
            
            if agent_result.get("task_successful", False):
                # Add successful results to accumulated context
                accumulated_context.update(agent_result.get("final_result", {}))
            else:
                logger.warning(f"Agent {agent.name} subtask failed, continuing with available results")
        
        return {
            "coordination_type": "sequential",
            "agents_used": [agent.name for agent in agents],
            "sequential_results": results,
            "final_accumulated_context": accumulated_context
        }
    
    async def _execute_parallel_coordination(self, task: MultiAgentTask, agents: List[IntelligentAgent], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute parallel coordination where agents work simultaneously"""
        
        logger.info(f"Executing parallel coordination with {len(agents)} agents")
        
        # Prepare tasks for all agents
        agent_tasks = []
        for i, (agent, subtask) in enumerate(zip(agents, strategy.get("subtasks", []))):
            agent_task = {
                "type": "parallel_subtask",
                "subtask_id": subtask.get("id", f"subtask_{i+1}"),
                "description": subtask.get("description", ""),
                "input_data": task.input_data,
                "objective": task.objective,
                "coordination_type": "parallel",
                "agent_index": i,
                "total_agents": len(agents)
            }
            agent_tasks.append((agent, agent_task))
        
        # Execute all tasks in parallel
        parallel_results = await asyncio.gather(
            *[agent.process_task(task_data) for agent, task_data in agent_tasks],
            return_exceptions=True
        )
        
        # Organize results
        results = {}
        for i, (agent, result) in enumerate(zip(agents, parallel_results)):
            if isinstance(result, Exception):
                results[f"agent_{i+1}_{agent.name}"] = {
                    "task_successful": False,
                    "error": str(result)
                }
            else:
                results[f"agent_{i+1}_{agent.name}"] = result
        
        return {
            "coordination_type": "parallel",
            "agents_used": [agent.name for agent in agents],
            "parallel_results": results
        }
    
    async def _execute_consensus_coordination(self, task: MultiAgentTask, agents: List[IntelligentAgent], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute consensus coordination where multiple agents solve the same problem"""
        
        logger.info(f"Executing consensus coordination with {len(agents)} agents")
        
        # All agents work on the same task
        consensus_task = {
            "type": "consensus_task",
            "description": task.description,
            "input_data": task.input_data,
            "objective": task.objective,
            "coordination_type": "consensus",
            "validation_required": True
        }
        
        # Execute task with all agents
        consensus_results = await asyncio.gather(
            *[agent.process_task(consensus_task) for agent in agents],
            return_exceptions=True
        )
        
        # Analyze consensus
        valid_results = [
            result for result in consensus_results 
            if not isinstance(result, Exception) and result.get("task_successful", False)
        ]
        
        # Use LLM to determine consensus
        consensus_analysis = await self._analyze_consensus(task, valid_results, agents)
        
        results = {}
        for i, (agent, result) in enumerate(zip(agents, consensus_results)):
            if isinstance(result, Exception):
                results[f"agent_{i+1}_{agent.name}"] = {
                    "task_successful": False,
                    "error": str(result)
                }
            else:
                results[f"agent_{i+1}_{agent.name}"] = result
        
        return {
            "coordination_type": "consensus",
            "agents_used": [agent.name for agent in agents],
            "consensus_results": results,
            "consensus_analysis": consensus_analysis,
            "agreed_result": consensus_analysis.get("agreed_result", {})
        }
    
    async def _execute_default_coordination(self, task: MultiAgentTask, agents: List[IntelligentAgent], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Default coordination fallback"""
        return await self._execute_sequential_coordination(task, agents, strategy)
    
    async def _analyze_consensus(self, task: MultiAgentTask, results: List[Dict[str, Any]], agents: List[IntelligentAgent]) -> Dict[str, Any]:
        """Use LLM to analyze consensus among agent results"""
        
        messages = [
            LLMMessage(role="system", content=self.system_prompt),
            LLMMessage(
                role="user",
                content=f"""CONSENSUS ANALYSIS:

Original Task: {task.description}
Objective: {task.objective}

Agent Results: {json.dumps(results, indent=2)}

Agents: {[agent.name for agent in agents]}

Please analyze the consensus among these results:
1. What do the agents agree on?
2. Where do they disagree?
3. Which result appears most accurate/complete?
4. What should be the final consensus result?
5. What confidence level should we assign?

Provide a synthesis that represents the best consensus view."""
            )
        ]
        
        response = await self.llm_engine.complete(
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        
        return {
            "consensus_analysis": response.content,
            "agreement_level": len(results) / len(agents) if agents else 0,
            "agreed_result": results[0] if results else {}  # Simplified - would be more sophisticated
        }
    
    async def _synthesize_multi_agent_results(self, task: MultiAgentTask, execution_result: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to synthesize final results from multiple agents"""
        
        messages = [
            LLMMessage(role="system", content=self.system_prompt),
            LLMMessage(
                role="user",
                content=f"""MULTI-AGENT RESULT SYNTHESIS:

Original Task:
- Description: {task.description}
- Objective: {task.objective}
- Input Data: {json.dumps(task.input_data, indent=2)}

Coordination Strategy: {strategy['type']}

Execution Results: {json.dumps(execution_result, indent=2)}

Please synthesize these multi-agent results:
1. Did we achieve the original objective?
2. What is the final answer/solution?
3. What evidence supports this conclusion?
4. What were the key insights from agent collaboration?
5. How confident are we in this result?
6. What recommendations do you have?

Provide a comprehensive final result that represents the best outcome from multi-agent collaboration."""
            )
        ]
        
        response = await self.llm_engine.complete(
            messages=messages,
            temperature=0.3,
            max_tokens=2000
        )
        
        # Calculate overall success
        coordination_type = execution_result.get("coordination_type", "unknown")
        
        if coordination_type == "sequential":
            agent_results = execution_result.get("sequential_results", {})
        elif coordination_type == "parallel":
            agent_results = execution_result.get("parallel_results", {})
        elif coordination_type == "consensus":
            agent_results = execution_result.get("consensus_results", {})
        else:
            agent_results = {}
        
        successful_agents = sum(1 for result in agent_results.values() 
                              if result.get("task_successful", False))
        total_agents = len(agent_results)
        success_rate = successful_agents / max(total_agents, 1)
        
        # Free up agents
        for agent_name in task.assigned_agents:
            if agent_name in self.agent_status:
                self.agent_status[agent_name] = AgentStatus.IDLE
        
        return {
            "task_id": task.task_id,
            "objective_achieved": success_rate >= 0.7,
            "multi_agent_synthesis": response.content,
            "coordination_used": coordination_type,
            "agents_involved": task.assigned_agents,
            "success_rate": success_rate,
            "execution_details": execution_result,
            "final_confidence": min(success_rate, 0.95),
            "completion_timestamp": datetime.now().isoformat(),
            "orchestrator_assessment": response.content
        }
    
    def _update_completion_metrics(self, task: MultiAgentTask):
        """Update system performance metrics"""
        execution_time = (datetime.now() - task.created_at).total_seconds()
        
        # Update average completion time
        current_avg = self.metrics["avg_completion_time"]
        total_completed = self.metrics["completed_tasks"]
        self.metrics["avg_completion_time"] = (current_avg * (total_completed - 1) + execution_time) / total_completed
        
        # Update agent utilization
        total_agents = len(self.agents)
        busy_agents = sum(1 for status in self.agent_status.values() if status == AgentStatus.BUSY)
        self.metrics["agent_utilization"] = busy_agents / max(total_agents, 1)
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a specific task"""
        
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "task_id": task_id,
                "status": "active",
                "progress": "in_progress",
                "assigned_agents": task.assigned_agents,
                "created_at": task.created_at.isoformat()
            }
        
        # Check completed tasks
        for task in self.completed_tasks:
            if task.task_id == task_id:
                return {
                    "task_id": task_id,
                    "status": task.status,
                    "results": task.results,
                    "assigned_agents": task.assigned_agents,
                    "created_at": task.created_at.isoformat()
                }
        
        # Check queue
        for task in self.task_queue:
            if task.task_id == task_id:
                return {
                    "task_id": task_id,
                    "status": "queued",
                    "position_in_queue": self.task_queue.index(task) + 1,
                    "created_at": task.created_at.isoformat()
                }
        
        return {"error": f"Task {task_id} not found"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "orchestrator_id": f"orchestrator_{id(self)}",
            "system_metrics": self.metrics,
            "agents": {
                name: {
                    "status": self.agent_status[name].value,
                    "capabilities": self.agent_capabilities[name],
                    "performance": agent.metrics
                }
                for name, agent in self.agents.items()
            },
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "system_uptime": (datetime.now() - self.metrics["system_uptime"]).total_seconds(),
            "status_timestamp": datetime.now().isoformat()
        }

# Export
__all__ = ["IntelligentMultiAgentOrchestrator", "MultiAgentTask", "TaskPriority", "AgentStatus"]