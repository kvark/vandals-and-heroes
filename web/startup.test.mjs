import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { startGame } from './startup.mjs';

const html = await readFile(new URL('./index.html', import.meta.url), 'utf8');

class Element {
    hidden;
    textContent = '';
    focused = false;
    listeners = new Map();
    classes = new Set();
    classList = {
        add: name => this.classes.add(name),
        remove: name => this.classes.delete(name),
        contains: name => this.classes.has(name),
    };
    constructor(hidden = false) { this.hidden = hidden; }
    focus() { this.focused = true; }
    addEventListener(type, listener) {
        const listeners = this.listeners.get(type) ?? [];
        listeners.push(listener);
        this.listeners.set(type, listeners);
    }
    emit(type, event = {}) {
        for (const listener of this.listeners.get(type) ?? []) listener(event);
    }
}

function page({ webgl = true, contextThrows = false, loadGame, paint } = {}) {
    // Build the minimal DOM from the real page, so renamed/missing element IDs
    // and initial visibility are covered along with the startup state machine.
    const elements = new Map([...html.matchAll(/<\w+\b[^>]*\bid="([^"]+)"[^>]*>/g)]
        .map(match => [match[1], new Element(/\bhidden\b/.test(match[0]))]));
    const events = [];
    const errors = [];
    const window = new Element();
    window.console = { error: error => errors.push(error) };
    window.location = { reload: () => events.push('reload') };
    const graphics = {
        getExtension: name => {
            assert.equal(name, 'WEBGL_lose_context');
            return { loseContext: () => events.push('release probe') };
        },
    };
    const document = {
        getElementById: id => {
            assert.ok(elements.has(id), `Missing element #${id}`);
            return elements.get(id);
        },
        createElement: tag => {
            assert.equal(tag, 'canvas');
            return {
                getContext: (type, options) => {
                    assert.equal(type, 'webgl2');
                    assert.deepEqual(options, { antialias: false });
                    if (contextThrows) throw new Error('WebGL blocked');
                    return webgl ? graphics : null;
                },
            };
        },
    };
    return {
        elements, events, errors, window,
        run: () => startGame({
            document, window,
            loadGame: loadGame ?? (async () => {
                events.push('import');
                return { default: async () => events.push('init') };
            }),
            paint: paint ?? (async () => events.push('paint')),
        }),
    };
}

function assertFailure(p, text) {
    assert.equal(p.elements.get('overlay').classList.contains('hidden'), false);
    assert.equal(p.elements.get('overlay').classList.contains('error'), true);
    assert.match(p.elements.get('status').textContent, text);
    assert.equal(p.elements.get('retry').hidden, false);
    assert.equal(p.elements.get('retry').focused, true);
    assert.equal(p.elements.get('game-status').hidden, true);
    assert.equal(p.elements.get('controls').hidden, true);
}

function assertReady(p) {
    assert.equal(p.elements.get('overlay').classList.contains('hidden'), true);
    assert.equal(p.elements.get('overlay').classList.contains('error'), false);
    assert.equal(p.elements.get('retry').hidden, true);
    assert.equal(p.elements.get('game-status').hidden, false);
    assert.equal(p.elements.get('controls').hidden, false);
    assert.equal(p.elements.get('blade').focused, true);
}

test('HTML includes the status HUD, missing gameplay keys, and startup module', () => {
    assert.match(html, /id="game-status" role="status" aria-live="polite" hidden/);
    assert.match(html, /V trigger chase/);
    assert.match(html, /F fire spike \(after depot\)/);
    assert.match(html, /Hold\/release Space/);
    assert.match(html, /import \{ startGame \} from '\.\/startup\.mjs'/);
});

test('HUD and controls wait for startup, then appear with canvas focus', async () => {
    const p = page();
    assert.equal(p.elements.get('game-status').hidden, true);
    assert.equal(p.elements.get('controls').hidden, true);
    await p.run();
    assertReady(p);
    assert.deepEqual(p.events, ['release probe', 'import', 'paint', 'init']);
});

for (const contextThrows of [false, true]) {
    test(`unavailable WebGL2 gives advice before downloading wasm (throws=${contextThrows})`, async () => {
        const p = page({ webgl: false, contextThrows });
        await p.run();
        assertFailure(p, /WebGL2 is unavailable/);
        assert.match(p.elements.get('startup-help').textContent, /hardware acceleration/);
        assert.deepEqual(p.events, []);
        p.elements.get('retry').emit('click');
        assert.deepEqual(p.events, ['reload']);
    });
}

test('winit control-flow exception counts as successful startup', async () => {
    const p = page({ loadGame: async () => ({
        default: () => { throw new Error('Using exceptions for control flow, don\'t mind me.'); },
    }) });
    await p.run();
    assertReady(p);
    assert.deepEqual(p.errors, []);
});

for (const stage of ['import', 'init']) {
    test(`${stage} failure offers reload and logs diagnostics without showing a raw panic`, async () => {
        const error = new Error('unreachable: internal panic details');
        const p = page({ loadGame: async () => {
            if (stage === 'import') throw error;
            return { default: () => { throw error; } };
        } });
        await p.run();
        assertFailure(p, /could not start/);
        assert.doesNotMatch(p.elements.get('status').textContent, /unreachable/);
        assert.deepEqual(p.errors, [error]);
    });
}

test('a context loss during initialization cannot be hidden by successful init', async () => {
    const p = page({ loadGame: async () => ({ default: () => {
        p.elements.get('blade').emit('webglcontextlost');
    } }) });
    await p.run();
    assertFailure(p, /graphics connection was lost/);
    assert.equal(p.elements.get('blade').focused, false);
});

test('an error while waiting to paint aborts init', async () => {
    let initialized = false;
    const p = page({
        loadGame: async () => ({ default: () => { initialized = true; } }),
        paint: async () => p.window.emit('error', { message: 'late failure' }),
    });
    await p.run();
    assertFailure(p, /could not start/);
    assert.equal(initialized, false);
});

test('context loss after startup restores the overlay and keeps the first useful error', async () => {
    const p = page();
    await p.run();
    p.elements.get('blade').emit('webglcontextlost');
    assertFailure(p, /graphics connection was lost/);
    p.window.emit('error', { error: new Error('secondary renderer panic') });
    assertFailure(p, /graphics connection was lost/);
    p.elements.get('retry').emit('click');
    assert.equal(p.events.filter(event => event === 'reload').length, 1);
});

for (const event of ['error', 'unhandledrejection']) {
    test(`${event} after startup exposes recovery instead of leaving a frozen canvas`, async () => {
        const p = page();
        await p.run();
        const error = new Error('later failure');
        p.window.emit(event, { error, reason: error });
        assertFailure(p, /stopped unexpectedly/);
        assert.deepEqual(p.errors, [error]);
    });
}

test('an asynchronous winit control-flow exception does not report a crash', async () => {
    const p = page();
    await p.run();
    p.window.emit('error', { error: new Error('Using exceptions for control flow') });
    assertReady(p);
    assert.deepEqual(p.errors, []);
});

// Compile-only wasm checks do not catch std clock panics. Guard every radio
// callsite, including startup chatter and the less frequent combat/reward paths.
test('game radio uses browser-safe wall clocks at every callsite', async () => {
    const game = await readFile(new URL('../bin/game/main.rs', import.meta.url), 'utf8');
    assert.match(game, /use web_time as time;/);
    assert.doesNotMatch(game, /std::time::(?:SystemTime|Instant|UNIX_EPOCH)/);
    assert.match(game, /time::SystemTime::now\(\)/);
    assert.match(game, /duration_since\(time::UNIX_EPOCH\)/);
});
