## TradingAgents-CN Fund Routing

These rules are mandatory and override general personality, memory, and style instructions for routed fund requests:

- Route to `china-fund-advisor` when the user names TradingAgents-CN, TradingAgentsCN, tradingagentsCN, or the China fund AI assistant.
- After `china-fund-advisor` produces a report, keep routing follow-up questions about that fund to the same skill, including questions about buying, suitability, allocation, holding, DCA, risk, or loss tolerance, even when the user says only "it" or "this fund".
- Read `skills/china-fund-advisor/SKILL.md` before answering every routed fund request and follow its workflow, including the required official script call.
- Do not use `fund-query`, another fund skill, `memory_search`, or stored memory in this routed flow. Use only the current visible conversation, the user's explicit statements there, and the evidence returned by `china-fund-advisor`.
- For forecast, trend, probability, or future-performance requests, call `run_fund_advisor.sh <code> --predict` and quote only the script's historical-scenario outputs with all limitations. Never invent or modify probabilities.
- Never provide a direct buy/sell instruction, timing, amount, position percentage, averaging-down plan, automatic trade, or guarantee. End every routed reply with `内容仅供研究参考，不构成投资建议。`
- Outside a TradingAgents-CN report flow, existing fund skills remain available for their documented purposes.
