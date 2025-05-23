package dev.langchain4j.agentic.internal;

import dev.langchain4j.agentic.Agent;
import dev.langchain4j.agentic.Cognisphere;
import dev.langchain4j.service.MemoryId;
import java.lang.annotation.Annotation;
import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

import static dev.langchain4j.internal.Utils.getAnnotatedMethod;
import static dev.langchain4j.internal.Utils.isNullOrBlank;

public class AgentUtil {

    private AgentUtil() { }

    public static List<AgentExecutor> agentsToExecutors(Object... agents) {
        return Stream.of(agents).map(AgentUtil::agentToExecutor).toList();
    }

    public static AgentExecutor agentToExecutor(Object agent) {
        if (agent instanceof AgentInstance) {
            return agentToExecutor((AgentInstance) agent);
        }
        Method agenticMethod = validateAgentClass(agent.getClass());
        Agent annotation = agenticMethod.getAnnotation(Agent.class);
        String name = isNullOrBlank(annotation.name()) ? agenticMethod.getName() : annotation.name();
        return new AgentExecutor(AgentSpecification.fromMethodAndSpec(agenticMethod, name, annotation.value(), annotation.outputName()), agent);
    }

    public static AgentExecutor agentToExecutor(AgentInstance agent) {
       for (Method method : agent.getClass().getDeclaredMethods()) {
           Optional<AgentExecutor> executor = agent instanceof A2AClientInstance a2a ?
                   methodToA2AExecutor(a2a, method) :
                   methodToAgentExecutor(agent, method);
           if (executor.isPresent()) {
                return executor.get();
           }
        }
        throw new IllegalArgumentException("Agent not found");
    }

    public static Optional<Method> getAnnotatedMethodOnClass(Class<?> clazz, Class<? extends Annotation> annotation) {
        return Arrays.stream(clazz.getDeclaredMethods())
                .filter(m -> m.isAnnotationPresent(annotation))
                .findFirst();
    }

    private static Optional<AgentExecutor> methodToA2AExecutor(A2AClientInstance a2aClient, Method method) {
        return getAnnotatedMethod(method, Agent.class)
                .map(agentMethod -> new AgentExecutor(new A2AClientAgentSpecification(a2aClient, agentMethod), a2aClient));
    }

    private static Optional<AgentExecutor> methodToAgentExecutor(AgentInstance agent, Method method) {
        return getAnnotatedMethod(method, Agent.class)
                .map(agentMethod -> new AgentExecutor(AgentSpecification.fromMethod(agent, agentMethod), agent));
    }

    public static Object[] methodInvocationArguments(Cognisphere cognisphere, Method method) {
        Parameter[] parameters = method.getParameters();
        if (parameters.length == 1) {
            return singleParamArg(cognisphere, parameters[0]);
        }

        Object[] invocationArgs = new Object[parameters.length];
        int i = 0;
        for (Parameter parameter : parameters) {
            if (parameter.getAnnotation(MemoryId.class) != null) {
                invocationArgs[i++] = cognisphere.id();
                continue;
            }
            String argName = AgentSpecification.parameterName(parameter);
            Object argValue = cognisphere.readState(argName);
            if (argValue == null) {
                throw new IllegalArgumentException("Missing argument: " + argName);
            }
            invocationArgs[i++] = parseArgument(argValue, parameter.getType());
        }
        return invocationArgs;
    }

    static Object[] singleParamArg(Cognisphere cognisphere, Parameter parameter) {
        if (parameter.getAnnotation(MemoryId.class) != null) {
            return new Object[]{cognisphere.id()};
        }
        Object argValue = AgentSpecification.optionalParameterName(parameter)
                .map(cognisphere::readState)
                .orElseGet(() -> cognisphere.getState().values().iterator().next());
        return new Object[]{parseArgument(argValue, parameter.getType())};
    }

    static Object parseArgument(Object argValue, Class<?> type) {
        if (argValue instanceof String s) {
            return switch (type.getName()) {
                case "java.lang.String" -> s;
                case "int", "java.lang.Integer" -> Integer.parseInt(s);
                case "long", "java.lang.Long" -> Long.parseLong(s);
                case "double", "java.lang.Double" -> Double.parseDouble(s);
                case "boolean", "java.lang.Boolean" -> Boolean.parseBoolean(s);
                default -> throw new IllegalArgumentException("Unsupported type: " + type);
            };
        }
        return argValue;
    }

    public static Method validateAgentClass(Class<?> agentServiceClass) {
        Method agentMethod = null;
        for (Method method : agentServiceClass.getDeclaredMethods()) {
            if (method.isAnnotationPresent(Agent.class)) {
                if (agentMethod != null) {
                    throw new IllegalArgumentException("Multiple agent methods found in class: " + agentServiceClass.getName());
                }
                agentMethod = method;
            }
        }
        if (agentMethod == null) {
            throw new IllegalArgumentException("No agent method found in class: " + agentServiceClass.getName());
        }
        return agentMethod;
    }
}
