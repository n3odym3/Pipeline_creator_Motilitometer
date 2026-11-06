# AI Helper Module

The **AI Helper** is a developer assistance module that integrates with local LLM servers (like **LM Studio**) to let you query, analyze, and explain module source code or get codebase summaries directly within Pipeline Creator.

---

## Key Features

- 🤖 **Local LLM Integration**: Connects to any local inference server exposing an OpenAI-compatible API (defaults to LM Studio at `http://localhost:1234`).
- 📂 **Contextual Module Selection**: Pick any discovered module in the project or load a full **Codebase Summary** to send as context with your prompt.
- ⚡ **Real-Time Streaming**: Watch responses stream into the UI character-by-character.
- 🎨 **Markdown Renderer**: Formats generated responses directly inside DearPyGui with headers, bullets, bold text, and code block styling.

---

## Configuration & Usage

1. **Start your Local Server**: Launch LM Studio (or Ollama/LocalAI) and start the local server on `http://localhost:1234`. Make sure a model is loaded.
2. **Add the Node**: In the Node Editor, right-click and add the **AI Helper** node under the `ai_helper` category.
3. **Configure Connection**:
   - **URL**: Set the API endpoint (e.g. `http://localhost:1234`).
   - **Model**: Select the active model name, or click the **Refresh** icon to query loaded models from the server.
4. **Choose Context**:
   - Select a specific module class from the dropdown to focus the LLM's attention on its source code, or select `[Codebase Summary]` to ask questions about the overall architecture.
5. **Ask & Send**:
   - Type your question in the text box and click **Send**. The response will stream in real time.
