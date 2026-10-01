// The control socket's operations: what each MCP tool (tools/local_llm_tool.mjs)
// and the chat responder reach through AgentServer, mapped onto the controller
// (DS001). main.mjs serves exactly this map; tests drive the same map.

export function controllerHandlers(controller) {
    return {
        overview: (args) => controller.overview(args),
        status: (args) => controller.status(args),
        run: (args) => controller.run(args),
        stop: () => controller.stop(),
        cancelDownload: () => controller.cancelDownload(),
        deleteWeights: (args) => controller.deleteWeights(args),
        addModel: (args) => controller.addModel(args.model),
        lookupModel: (args) => controller.lookupModel(args),
        updateModel: (args) => controller.updateModel(args.model),
        removeModel: (args) => controller.removeModel(args),
        installRunner: (args) => controller.installRunner(args),
        uninstallRunner: (args) => controller.uninstallRunner(args),
        chatTarget: () => controller.chatTarget(),
        recordCompletion: (args) => controller.recordCompletion(args),
    };
}
