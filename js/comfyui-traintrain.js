import { app } from "../../scripts/app.js";
import { ComfyButtonGroup } from "../../scripts/ui/components/buttonGroup.js";
import { ComfyButton } from "../../scripts/ui/components/button.js";

app.registerExtension({
    name: "love2hina.ComfyUI.traintrain",
    init() {},
    setup() {
        const btnTrain = new ComfyButton({
            content: "TrainTrain",
            tooltip: "",
            classList: "comfyui-button comfyui-menu-mobile-collapse primary",
            action: () => {}
        });

        app.menu.settingsGroup.element.before((new ComfyButtonGroup(btnTrain.element)).element);
    },
})
