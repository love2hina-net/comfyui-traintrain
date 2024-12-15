import { app } from "../../scripts/app.js";
import { $el, ComfyDialog } from "../../scripts/ui.js";
import { ComfyButtonGroup } from "../../scripts/ui/components/buttonGroup.js";
import { ComfyButton } from "../../scripts/ui/components/button.js";

class TrainDialog extends ComfyDialog {
    constructor() {
        super();

        const content = $el("div.comfy-modal-content", {}, [
            $el("tr.cm-title", {}, [
                $el("font", {size:6, color:"white"}, [`TrainTrain`])
            ]),
            $el("br", {}, []),
            $el("button", { id: "train-close-button", type: "button", textContent: "Close", onclick: () => this.close() })
        ]);
        content.style.width = '100%';
		content.style.height = '100%';

        this.element = $el("div.comfy-modal", { id: "train-dialog", parent: document.body }, [ content ]);
    }

    show() {
		this.element.style.display = "block";
	}
}

let dialog;

app.registerExtension({
    name: "love2hina.ComfyUI.traintrain",
    init() {},
    setup() {
        const btnTrain = new ComfyButton({
            content: "TrainTrain",
            tooltip: "",
            classList: "comfyui-button comfyui-menu-mobile-collapse primary",
            action: () => {
                if (!dialog) {
                    dialog = new TrainDialog();
                }
                dialog.show();
            }
        });

        app.menu.settingsGroup.element.before((new ComfyButtonGroup(btnTrain.element)).element);
    },
})
