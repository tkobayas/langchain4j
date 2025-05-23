package dev.langchain4j.agentic;

public record ResultWithCognisphere<T>(Cognisphere cognisphere, T result) { }
