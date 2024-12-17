import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";
import { $el, ComfyDialog } from "../../scripts/ui.js";
import { ComfyButtonGroup } from "../../scripts/ui/components/buttonGroup.js";
import { ComfyButton } from "../../scripts/ui/components/button.js";

class TrainDialog extends ComfyDialog {
    main_frame = null;

    constructor() {
        super();

        this.main_frame = $el("iframe", {style: {flex: "auto"}}, []);

        const content = $el(
            "div.comfy-modal-content",
            {
                style: {
                    display: "flex",
                    flexDirection: "column",
                    width: "100%",
                    height: "100%",
                }
            },
            [
                $el("tr.cm-title", {style: {flex: "none" }}, [
                    $el("font", {size: 6, color: "white"}, [`TrainTrain`])
                ]),
                this.main_frame,
                $el("button", {id: "train-close-button", type: "button", textContent: "Close", style: {flex: "none"}, onclick: () => this.close()})
            ]);

        this.element = $el(
            "div.comfy-modal",
            {
                id: "train-dialog",
                parent: document.body,
                style: {
                    width: "95%",
                    maxWidth: "95vw",
                    height: "95%",
                    maxHeight: "95vh",
                }
            },
            [ content ]);
    }

    async show() {
		this.element.style.display = "block";

        let response = await api.fetchApi('/traintrain/start_server');
        response.json().then(data => {
            this.main_frame.src = data.url;
        });
	}

    close() {
        super.close();

        api.fetchApi('/traintrain/stop_server');
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
