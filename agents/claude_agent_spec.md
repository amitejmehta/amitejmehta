# Claude Agent Specification

This document outlines the requirements for building agents that can serve as alternatives to or implementations of Claude-style AI assistants.

## General Agent Requirements

### Core Functionality

1. **Agent Loop with Tool Use**
   - Must implement a conversation loop that can handle multi-turn interactions
   - Support for function/tool calling with structured input/output
   - Ability to execute tools and incorporate results back into the conversation flow
   - Asynchronous execution support for concurrent tool calls

2. **Tracing and Observability**
   - Comprehensive tracing of all LLM interactions
   - Tool execution logging and timing
   - Error tracking and debugging capabilities
   - This project uses logfire for automatic instrumentation
     ```python
     logfire.configure(scrubbing=False, environment=runtime_env)
     logfire.instrument_anthropic()  # Single line integration
     logfire.instrument_openai()
     ```

### Optimization Features

3. **Automatic Tool Schema Generation**
   - Generate JSON schemas from Python function signatures
   - Type hint support for automatic validation
   - Documentation extraction from docstrings
   - Runtime validation of tool inputs/outputs

4. **Type-Safe Dependency Injection**
   - Tools must be able to accept dependencies in a type-safe manner
   - Context management for shared resources (API clients, database connections, etc.)
   - Lifecycle management for resources
   - **Implementation**: Use Context and ContextWrapper pattern from `context.py`
     ```python
     def tool_function(input: str, ctx: ContextWrapper) -> str:
         client = ctx.get_client()  # Type-safe access
         return client.do_something(input)
     ```

## Claude Code Agent Requirements

A Claude Code agent extends the general agent requirements with specific tooling for code-related tasks.

### Required Tools

In addition to general agent requirements, a Claude Code agent must provide:

1. **bash** - Execute shell commands with proper error handling and security constraints
2. **read_file** - Read file contents with path validation and encoding detection
3. **edit_file** - Modify files with atomic operations and backup capabilities

### CLI Experience Requirements

For a solid command-line interface experience:

4. **Readable Output Display**
   - Syntax highlighting for code blocks
   - Structured formatting for tool outputs
   - Progress indicators for long-running operations
   - Clear distinction between agent responses and tool outputs
   - **Implementation**: Use Rich console for formatting

5. **Multi-line Input Support**
   - Handle pasted code blocks without treating each line as separate input
   - Support for heredoc-style input
   - Ability to enter "paste mode" for large text blocks
   - Preserve formatting and indentation

### UI Features

6. **Slash Commands**
   - `/clear` - Clear message history
   - `/help` - Show available commands
   - `/paste` - Enter multi-line paste mode
   - Extensible command system for additional functionality

7. **Session Management**
   - Persistent conversation history
   - Ability to save/load sessions
   - Context preservation across restarts

## Implementation Notes

### Architecture Patterns

- **Agent Factory Pattern**: Use factory functions to create agents with different configurations
- **Tool Registration**: Dynamic tool registration and discovery
- **Context Isolation**: Each agent instance should have isolated context
- **Resource Cleanup**: Proper cleanup of resources on agent shutdown

### Security Considerations

- Sandbox shell commands where possible
- File system access restrictions
- Input validation for all tools
- Rate limiting for API calls
- Audit logging for sensitive operations

### Performance Requirements

- Tool calls should execute concurrently when possible
- Response streaming for real-time feedback
- Caching frequently used data
- Efficient message history management

## Example Implementation Structure

```python
class ClaudeCodeAgent:
    def __init__(self, context: Context):
        self.llm = LLM(model="claude-3-sonnet", context=context)
        self.tools = [bash, read_file, edit_file]
        self.context = context
    
    async def chat(self, message: str) -> str:
        return await self.llm.loop(message, tools=self.tools)
    
    def handle_slash_command(self, command: str) -> None:
        if command == "/clear":
            self.llm.clear_history()
        # Handle other commands...
```

This specification ensures that any Claude agent implementation provides consistent functionality while allowing for customization and extension based on specific use cases.