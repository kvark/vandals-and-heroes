// Kept separate from the wasm module so startup failures can be tested without
// a GPU, downloading the assets, or starting winit's event loop.
export async function startGame({
    document = globalThis.document,
    window = globalThis.window,
    loadGame = () => import('./pkg/game.js'),
    paint = () => new Promise(resolve => window.requestAnimationFrame(() => window.setTimeout(resolve))),
} = {}) {
    const status = document.getElementById('status');
    const overlay = document.getElementById('overlay');
    const help = document.getElementById('startup-help');
    const retry = document.getElementById('retry');
    const canvas = document.getElementById('blade');
    const hud = document.getElementById('game-status');
    const controls = document.getElementById('controls');
    let failed = false;
    let started = false;

    const isEventLoopStart = error =>
        /Using exceptions for control flow/.test(String(error));

    function fail(message, guidance, error) {
        // Keep the useful first error, rather than replacing e.g. context-loss
        // guidance with a secondary rendering panic.
        if (failed) return;
        failed = true;
        overlay.classList.remove('hidden');
        overlay.classList.add('error');
        status.textContent = message;
        help.textContent = guidance;
        hud.hidden = true;
        controls.hidden = true;
        retry.hidden = false;
        retry.focus();
        if (error) window.console.error(error);
    }

    function ready() {
        // A context-loss or asynchronous error may have arrived during init.
        if (failed) return;
        started = true;
        overlay.classList.add('hidden');
        hud.hidden = false;
        controls.hidden = false;
        canvas.focus();
    }

    function reportError(error) {
        // winit deliberately throws this when it hands control to the browser.
        if (isEventLoopStart(error)) return;
        fail(
            started ? 'The game stopped unexpectedly.' : 'The game could not start.',
            'Reload to try again. If it keeps failing, check the browser console for details.',
            error,
        );
    }

    retry.addEventListener('click', () => window.location.reload());
    window.addEventListener('error', event => reportError(event.error ?? event.message));
    window.addEventListener('unhandledrejection', event => reportError(event.reason));
    canvas.addEventListener('webglcontextlost', () => fail(
        'The graphics connection was lost.',
        'Reload to restart the game. Close other graphics-heavy tabs if this keeps happening.',
    ));

    // Probe a separate canvas: probing #blade would fix its context options
    // before Blade can choose them. Release the probe's GPU resources promptly.
    let graphics;
    try {
        graphics = document.createElement('canvas').getContext('webgl2', { antialias: false });
    } catch {
        // Some browsers throw instead of returning null when WebGL is blocked.
    }
    if (!graphics) {
        fail(
            'WebGL2 is unavailable in this browser.',
            'Enable hardware acceleration in your browser settings, then reload. If needed, try another browser or device with WebGL2 support.',
        );
        return;
    }
    graphics.getExtension('WEBGL_lose_context')?.loseContext();

    try {
        status.textContent = 'Loading the game…';
        const { default: init } = await loadGame();
        if (failed) return;
        status.textContent = 'Starting the world… (the tab may pause)';
        // Paint the message before wasm initialization can synchronously build
        // the terrain and block the main thread.
        await paint();
        if (failed) return;
        await init();
        ready();
    } catch (error) {
        if (isEventLoopStart(error)) {
            ready();
        } else {
            reportError(error);
        }
    }
}
