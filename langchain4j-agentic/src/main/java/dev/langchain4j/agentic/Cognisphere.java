package dev.langchain4j.agentic;

import dev.langchain4j.agentic.internal.AgentCall;
import dev.langchain4j.agentic.internal.AgentSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.service.memory.ChatMemoryAccess;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Predicate;

public class Cognisphere {

    private static final Logger LOG = LoggerFactory.getLogger(Cognisphere.class);

    public record AgentMessage(String agentName, ChatMessage message) {}

    private final Object id;
    private final Map<String, Object> state = new ConcurrentHashMap<>();
    private final Map<String, List<AgentCall>> agentsInvocations = new ConcurrentHashMap<>();
    private final List<AgentMessage> context = new ArrayList<>();

    public Cognisphere() {
        this(UUID.randomUUID().toString());
    }

    public Cognisphere(Object id) {
        this.id = id;
    }

    public Object id() {
        return id;
    }

    public void writeState(String key, Object value) {
        state.put(key, value);
    }

    public void writeStates(Map<String, Object> newState) {
        state.putAll(newState);
    }

    public Object readState(String key) {
        return state.get(key);
    }

    public <T> T readState(String key, T defaultValue) {
        return (T) state.getOrDefault(key, defaultValue);
    }

    public Map<String, Object> getState() {
        return state;
    }

    public void registerAgentCall(AgentSpecification agentSpecification, Object agent, Object[] input, Object response) {
        agentsInvocations.computeIfAbsent(agentSpecification.name(), name -> new ArrayList<>())
                .add(new AgentCall(agentSpecification.name(), input, response));
        registerContext(agentSpecification, agent, input, response);
    }

    private void registerContext(AgentSpecification agentSpecification, Object agent, Object[] input, Object response) {
        if (agent instanceof ChatMemoryAccess agentWithMemory) {
            ChatMemory chatMemory = agentWithMemory.getChatMemory(id);
            if (chatMemory != null) {
                List<ChatMessage> agentMessages = chatMemory.messages();
                for (int i = agentMessages.size() - 1; i >= 0; i--) {
                    if (agentMessages.get(i) instanceof UserMessage userMessage) {
                        // Only add to the cognisphere's context the last UserMessage ...
                        context.add(new AgentMessage(agentSpecification.name(), userMessage));
                        // ... and last AiMessage response, all other messages are local to the invoked agent internals
                        context.add(new AgentMessage(agentSpecification.name(), agentMessages.get(agentMessages.size() - 1)));
                        return;
                    }
                }
            }
        }
        if (agentSpecification.description() != null && !agentSpecification.description().isBlank()) {
            context.add(new AgentMessage(agentSpecification.name(), new UserMessage(agentSpecification.description())));
            context.add(new AgentMessage(agentSpecification.name(), AiMessage.aiMessage(response.toString())));
        }
    }

    public List<AgentMessage> context() {
        return context;
    }

    public String contextAsConversation(String... agentNames) {
        Predicate<String> agentFilter = agentNames != null && agentNames.length > 0 ?
                List.of(agentNames)::contains :
                agent -> true;

        StringBuilder sb = new StringBuilder();
        for (AgentMessage agentMessage : context) {
            if (!agentFilter.test(agentMessage.agentName())) {
                continue;
            }
            ChatMessage message = agentMessage.message();
            if (message instanceof UserMessage userMessage) {
                sb.append("User: ").append(userMessage.singleText()).append("\n");
            } else if (message instanceof AiMessage aiMessage) {
                sb.append(agentMessage.agentName()).append(" agent: ").append(aiMessage.text()).append("\n");
            }
        }

        String context = sb.toString();
        LOG.info("Cognisphere context as conversation: '{}'", context);
        return context;
    }

    public List<AgentCall> getAgentInvocations(String agentName) {
        return agentsInvocations.getOrDefault(agentName, List.of());
    }

    @Override
    public String toString() {
        return "Cognisphere{" +
                "id='" + id + '\'' +
                ", state=" + state +
                '}';
    }

    public static CognisphereRegistry registry() {
        return CognisphereRegistry.getInstance();
    }

    public static class CognisphereRegistry {

        private static final CognisphereRegistry INSTANCE = new CognisphereRegistry();

        private final Map<Object, Cognisphere> cognisphereMap = new ConcurrentHashMap<>();

        private CognisphereRegistry() {
            // Prevent instantiation
        }

        public static CognisphereRegistry getInstance() {
            return INSTANCE;
        }

        public void register(Cognisphere cognisphere) {
            cognisphereMap.put(cognisphere.id(), cognisphere);
        }

        public Cognisphere get(Object id) {
            return cognisphereMap.get(id);
        }

        public Cognisphere getOrCreate(Object id) {
            return cognisphereMap.computeIfAbsent(id, Cognisphere::new);
        }

        public boolean evict(Object id) {
            return cognisphereMap.remove(id) != null;
        }

        public void clear() {
            cognisphereMap.clear();
        }
    }
}
