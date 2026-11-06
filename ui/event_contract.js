(() => {
    const token = __CHRONOS_CONTRACT_TOKEN__;
    const nativeFetch = window.fetch.bind(window);
    let reloading = false;
    const isEvent = (url) => url.origin === location.origin &&
        /\/(?:queue\/join|run\/[^/]+|api\/[^/]+|call\/[^/]+)\/?$/.test(url.pathname);

    function reloadPage() {
        if (reloading) return;
        reloading = true;
        // GET uses no-store on the server: reload receives the new event graph.
        location.reload();
    }

    window.fetch = async (input, init) => {
        const url = new URL(input instanceof Request ? input.url : input, location.href);
        const method = (init?.method || (input instanceof Request ? input.method : "GET")).toUpperCase();
        if (method !== "POST" || !isEvent(url)) return nativeFetch(input, init);
        if (reloading) throw new DOMException("Chronos page is reloading", "AbortError");
        const headers = new Headers(init?.headers || (input instanceof Request ? input.headers : undefined));
        headers.set("X-Chronos-UI-Contract", token);
        const response = await nativeFetch(input, {...init, headers});
        if (response.status === 409) {
            const error = await response.clone().json().catch(() => ({}));
            if (error.code === "chronos_ui_outdated") reloadPage();
        }
        return response;
    };

    // Also synchronize idle tabs, including those restored from the back/forward cache.
    let checking = false;
    async function checkServer() {
        if (checking || reloading || document.hidden) return;
        checking = true;
        try {
            const base = (window.gradio_config?.root || location.origin + location.pathname).replace(/\/$/, "");
            const response = await nativeFetch(base + "/chronos-ui/contract", {cache: "no-store"});
            if (response.ok && (await response.json()).token !== token) reloadPage();
        } catch (_) {
            // A stopped/restarting server is retried by the next tick.
        } finally {
            checking = false;
        }
    }
    window.addEventListener("pageshow", checkServer);
    document.addEventListener("visibilitychange", checkServer);
    setInterval(checkServer, 3000);
})();
