import { animateCountUps } from './animate.js';

// Minimal state -> render loop, mirroring the prototype's DCLogic pattern
// (state + renderVals()) so each screen's translated logic reads the same way.
export class Screen {
  constructor(root, initialState) {
    this.root = root;
    this.state = initialState;
    this._statValues = {};
    this.render();
    // Delegated listeners on the persistent root survive each re-render,
    // since only its innerHTML is replaced.
    this.bind(this.root);
  }

  setState(patch) {
    this.state = { ...this.state, ...patch };
    this.render();
  }

  render() {
    this.root.innerHTML = this.template(this.renderVals());
    // Any element marked [data-stat] counts up from its previous value.
    this._statValues = animateCountUps(this.root, this._statValues);
  }

  // Override in subclasses.
  renderVals() { return {}; }
  template() { return ''; }
  bind() {}
}
