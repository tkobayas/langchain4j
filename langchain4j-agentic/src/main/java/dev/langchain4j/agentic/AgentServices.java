package dev.langchain4j.agentic;

import dev.langchain4j.agentic.declarative.ActivationCondition;
import dev.langchain4j.agentic.declarative.ConditionalAgent;
import dev.langchain4j.agentic.declarative.ExitCondition;
import dev.langchain4j.agentic.declarative.LoopAgent;
import dev.langchain4j.agentic.declarative.Output;
import dev.langchain4j.agentic.declarative.ParallelAgent;
import dev.langchain4j.agentic.declarative.SequenceAgent;
import dev.langchain4j.agentic.declarative.Subagent;
import dev.langchain4j.agentic.internal.AgentExecutor;
import dev.langchain4j.agentic.internal.AgentInstance;
import dev.langchain4j.agentic.internal.AgentSpecification;
import dev.langchain4j.agentic.supervisor.SupervisorAgent;
import dev.langchain4j.agentic.supervisor.SupervisorAgentService;
import dev.langchain4j.agentic.workflow.ConditionialAgentService;
import dev.langchain4j.agentic.workflow.LoopAgentService;
import dev.langchain4j.agentic.workflow.OutputtingService;
import dev.langchain4j.agentic.workflow.ParallelAgentService;
import dev.langchain4j.agentic.workflow.SequentialAgentService;
import dev.langchain4j.model.chat.ChatModel;
import io.a2a.A2A;
import io.a2a.spec.A2AClientError;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Stream;

import static dev.langchain4j.agentic.internal.AgentUtil.agentToExecutor;
import static dev.langchain4j.agentic.internal.AgentUtil.getAnnotatedMethodOnClass;
import static dev.langchain4j.agentic.internal.AgentUtil.methodInvocationArguments;
import static dev.langchain4j.agentic.internal.AgentUtil.validateAgentClass;
import static dev.langchain4j.internal.Utils.isNullOrBlank;

public class AgentServices {

    private AgentServices() { }

    public static <T> AgentBuilder<T> agentBuilder(Class<T> agentServiceClass) {
        return new AgentBuilder<>(agentServiceClass, validateAgentClass(agentServiceClass));
    }

    public static SequentialAgentService<UntypedAgent> sequenceBuilder() {
        return SequentialAgentService.builder();
    }

    public static <T> SequentialAgentService<T> sequenceBuilder(Class<T> agentServiceClass) {
        return SequentialAgentService.builder(agentServiceClass);
    }

    public static ParallelAgentService<UntypedAgent> parallelBuilder() {
        return ParallelAgentService.builder();
    }

    public static <T> ParallelAgentService<T> parallelBuilder(Class<T> agentServiceClass) {
        return ParallelAgentService.builder(agentServiceClass);
    }

    public static LoopAgentService<UntypedAgent> loopBuilder() {
        return LoopAgentService.builder();
    }

    public static <T> LoopAgentService<T> loopBuilder(Class<T> agentServiceClass) {
        return LoopAgentService.builder(agentServiceClass);
    }

    public static ConditionialAgentService<UntypedAgent> conditionalBuilder() {
        return ConditionialAgentService.builder();
    }

    public static <T> ConditionialAgentService<T> conditionalBuilder(Class<T> agentServiceClass) {
        return ConditionialAgentService.builder(agentServiceClass);
    }

    public static SupervisorAgentService<SupervisorAgent> supervisorBuilder() {
        return SupervisorAgentService.builder();
    }

    public static <T> SupervisorAgentService<T> supervisorBuilder(Class<T> agentServiceClass) {
        return SupervisorAgentService.builder(agentServiceClass);
    }

    public static A2AClientBuilder<UntypedAgent> a2aBuilder(String a2aServerUrl) {
        return a2aBuilder(a2aServerUrl, UntypedAgent.class);
    }

    public static <T> A2AClientBuilder<T> a2aBuilder(String a2aServerUrl, Class<T> agentServiceClass) {
        try {
            return new A2AClientBuilder(A2A.getAgentCard(a2aServerUrl), agentServiceClass);
        } catch (A2AClientError e) {
            throw new RuntimeException(e);
        }
    }

    public static <T> T createAgenticSystem(Class<T> agentServiceClass, ChatModel chatModel) {
        T agent = createComposedAgent(agentServiceClass, chatModel);

        if (agent == null) {
            throw new IllegalArgumentException("Provided class " + agentServiceClass.getName() + " is not an agent.");
        }

        return agent;
    }

    private static <T> T createComposedAgent(Class<T> agentServiceClass, ChatModel chatModel) {
        Optional<Method> sequenceMethod = getAnnotatedMethodOnClass(agentServiceClass, SequenceAgent.class);
        if (sequenceMethod.isPresent()) {
            return buildSequentialAgent(agentServiceClass, sequenceMethod.get(), chatModel);
        }

        Optional<Method> loopMethod = getAnnotatedMethodOnClass(agentServiceClass, LoopAgent.class);
        if (loopMethod.isPresent()) {
            return buildLoopAgent(agentServiceClass, loopMethod.get(), chatModel);
        }

        Optional<Method> conditionalMethod = getAnnotatedMethodOnClass(agentServiceClass, ConditionalAgent.class);
        if (conditionalMethod.isPresent()) {
            return buildConditionalAgent(agentServiceClass, conditionalMethod.get(), chatModel);
        }

        Optional<Method> parallelMethod = getAnnotatedMethodOnClass(agentServiceClass, ParallelAgent.class);
        if (parallelMethod.isPresent()) {
            return buildParallelAgent(agentServiceClass, parallelMethod.get(), chatModel);
        }

         return null;
    }

    private static <T> T buildSequentialAgent(Class<T> agentServiceClass, Method agentMethod, ChatModel chatModel) {
        SequenceAgent sequenceAgent = agentMethod.getAnnotation(SequenceAgent.class);
        var builder = sequenceBuilder(agentServiceClass)
                .subAgents(createSubagents(sequenceAgent.subagents(), chatModel));

        buildOutput(agentServiceClass, sequenceAgent.outputName(), builder);

        return builder.build();
    }

    private static <T> T buildLoopAgent(Class<T> agentServiceClass, Method agentMethod, ChatModel chatModel) {
        LoopAgent loopAgent = agentMethod.getAnnotation(LoopAgent.class);
        var builder = loopBuilder(agentServiceClass)
                .subAgents(createSubagents(loopAgent.subagents(), chatModel))
                .maxIterations(loopAgent.maxIterations());

        buildOutput(agentServiceClass, loopAgent.outputName(), builder);

        predicateMethod(agentServiceClass, method -> method.isAnnotationPresent(ExitCondition.class))
                .map(AgentServices::cognispherePredicate)
                .ifPresent(builder::exitCondition);

        return builder.build();
    }

    private static <T> T buildConditionalAgent(Class<T> agentServiceClass, Method agentMethod, ChatModel chatModel) {
        ConditionalAgent conditionalAgent = agentMethod.getAnnotation(ConditionalAgent.class);
        var builder = conditionalBuilder(agentServiceClass);

        buildOutput(agentServiceClass, conditionalAgent.outputName(), builder);

        for (Subagent subagent : conditionalAgent.subagents()) {
            predicateMethod(agentServiceClass, method -> {
                ActivationCondition activationCondition = method.getAnnotation(ActivationCondition.class);
                return activationCondition != null && Arrays.asList(activationCondition.value()).contains(subagent.agentName());
            })
                    .map(AgentServices::cognispherePredicate)
                    .ifPresent(condition -> builder.subAgent(condition, createSubagent(subagent, chatModel)));
        }

        return builder.build();
    }

    private static <T> T buildParallelAgent(Class<T> agentServiceClass, Method agentMethod, ChatModel chatModel) {
        ParallelAgent parallelAgent = agentMethod.getAnnotation(ParallelAgent.class);
        var builder = parallelBuilder(agentServiceClass)
                .subAgents(createSubagents(parallelAgent.subagents(), chatModel));

        buildOutput(agentServiceClass, parallelAgent.outputName(), builder);

        return builder.build();
    }

    private static <T> void buildOutput(Class<T> agentServiceClass, String outputName, OutputtingService<?> builder) {
        if (!outputName.isBlank()) {
            builder.outputName(outputName);
        }

        selectMethod(agentServiceClass, method -> method.isAnnotationPresent(Output.class))
                .map(AgentServices::cognisphereFunction)
                .ifPresent(builder::output);
    }

    private static Optional<Method> predicateMethod(Class<?> agentServiceClass, Predicate<Method> methodSelector) {
        return selectMethod(agentServiceClass, methodSelector.and(m -> (m.getReturnType() == boolean.class || m.getReturnType() == Boolean.class)));
    }

    private static Optional<Method> selectMethod(Class<?> agentServiceClass, Predicate<Method> methodSelector) {
        for (Method method : agentServiceClass.getDeclaredMethods()) {
            if (methodSelector.test(method) && Modifier.isStatic(method.getModifiers())) {
                return Optional.of(method);
            }
        }
        return Optional.empty();
    }

    private static Predicate<Cognisphere> cognispherePredicate(Method predicateMethod) {
        return cognisphere -> (boolean) cognisphereFunction(predicateMethod).apply(cognisphere);
    }

    private static Function<Cognisphere, Object> cognisphereFunction(Method functionMethod) {
        boolean isCognisphereArg = functionMethod.getParameterCount() == 1 && functionMethod.getParameterTypes()[0] == Cognisphere.class;
        return cognisphere -> {
            try {
                Object[] args = isCognisphereArg ? new Object[] {cognisphere} : methodInvocationArguments(cognisphere, functionMethod);
                return functionMethod.invoke(null, args);
            } catch (Exception e) {
                throw new RuntimeException("Error invoking exit condition method: " + functionMethod.getName(), e);
            }
        };
    }

    private static List<AgentExecutor> createSubagents(Subagent[] subagents, ChatModel chatModel) {
        return Stream.of(subagents)
                .map(subagent -> createSubagent(subagent, chatModel))
                .toList();
    }

    private static AgentExecutor createSubagent(Subagent subagent, ChatModel chatModel) {
        AgentExecutor agentExecutor = createComposedAgentExecutor(subagent.agentClass(), chatModel);
        if (agentExecutor != null) {
            return agentExecutor;
        }

        return agentToExecutor((AgentInstance) AgentServices.agentBuilder(subagent.agentClass())
                .chatModel(chatModel)
                .outputName(subagent.outputName())
                .build());
    }

    private static AgentExecutor createComposedAgentExecutor(Class<?> agentServiceClass, ChatModel chatModel) {
        Optional<Method> sequenceMethod = getAnnotatedMethodOnClass(agentServiceClass, SequenceAgent.class);
        if (sequenceMethod.isPresent()) {
            Method method = sequenceMethod.get();
            AgentInstance agent = (AgentInstance) buildSequentialAgent(agentServiceClass, method, chatModel);
            SequenceAgent annotation = method.getAnnotation(SequenceAgent.class);
            String name = annotation == null || isNullOrBlank(annotation.name()) ? method.getName() : annotation.name();
            String description = annotation == null ? "" : String.join("\n", annotation.value());
            return new AgentExecutor(AgentSpecification.fromMethodAndSpec(method, name, description, agent.outputName()), agent);
        }

        Optional<Method> loopMethod = getAnnotatedMethodOnClass(agentServiceClass, LoopAgent.class);
        if (loopMethod.isPresent()) {
            Method method = loopMethod.get();
            AgentInstance agent = (AgentInstance) buildLoopAgent(agentServiceClass, loopMethod.get(), chatModel);
            LoopAgent annotation = method.getAnnotation(LoopAgent.class);
            String name = annotation == null || isNullOrBlank(annotation.name()) ? method.getName() : annotation.name();
            String description = annotation == null ? "" : String.join("\n", annotation.value());
            return new AgentExecutor(AgentSpecification.fromMethodAndSpec(method, name, description, agent.outputName()), agent);
        }

        Optional<Method> conditionalMethod = getAnnotatedMethodOnClass(agentServiceClass, ConditionalAgent.class);
        if (conditionalMethod.isPresent()) {
            Method method = conditionalMethod.get();
            AgentInstance agent = (AgentInstance) buildConditionalAgent(agentServiceClass, conditionalMethod.get(), chatModel);
            ConditionalAgent annotation = method.getAnnotation(ConditionalAgent.class);
            String name = annotation == null || isNullOrBlank(annotation.name()) ? method.getName() : annotation.name();
            String description = annotation == null ? "" : String.join("\n", annotation.value());
            return new AgentExecutor(AgentSpecification.fromMethodAndSpec(method, name, description, agent.outputName()), agent);
        }

        Optional<Method> parallelMethod = getAnnotatedMethodOnClass(agentServiceClass, ParallelAgent.class);
        if (parallelMethod.isPresent()) {
            Method method = parallelMethod.get();
            AgentInstance agent = (AgentInstance) buildConditionalAgent(agentServiceClass, parallelMethod.get(), chatModel);
            ParallelAgent annotation = method.getAnnotation(ParallelAgent.class);
            String name = annotation == null || isNullOrBlank(annotation.name()) ? method.getName() : annotation.name();
            String description = annotation == null ? "" : String.join("\n", annotation.value());
            return new AgentExecutor(AgentSpecification.fromMethodAndSpec(method, name, description, agent.outputName()), agent);
        }

        return null;
    }
}
