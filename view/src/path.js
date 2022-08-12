class Node {
    constructor(parent, property, col, row) {

        this.parts = new Konva.Group();

        const rad = 12;
        const x = col * 48;
        const y = row * 36;

        const icon = new Konva.Circle({
            x: x,
            y: y,
            radius: rad,
            fill: "#fff",
            strokeWidth: 0,
            shadowBlur: 2,
            shadowOffset: { x: 2, y: 2 },
            shadowOpacity: 0.9,
        });

        const text = new Konva.Text({
            x: x - 9,
            y: y - 9,
            text: "" + property["selected"],
            fontSize: 18,
            fontFamily: "Calibri",
            fill: "#000",
            width: rad*2,
            padding: 0,
            align: "left",
            visible: true,
        });

        this.parts.add(icon);
        this.parts.add(text);
        parent.add(this.parts);
    }

    destroy() {
        this.parts.destroy();
    }
}


class Path_hierarchy {

    constructor(parent_ui, parent_bg, path_data, handle_selection) {
        console.log(path_data);
        this.handle_selection = handle_selection;

        this.nodes = new Konva.Group();

        this.step = 0;
        this.layers = {};
        this.stack = [];

        for (const item of path_data) {
            if (item["s"] != null) {
                this.layers[item["layer"]] = item;
                continue;
            }

            this._build(item);
            this.stack.push(item);
        }

        while (this.stack.length > 0) {
            const last = this.stack.pop();
            this._place(last);
        }


        parent_bg.add(this.nodes);
    }

    _place(item) {
        const item_layer = item["layer"];
        const node = new Node(this.nodes, item, this.step++, (Object.keys(this.layers).length - 1) - item_layer);
    }

    _build(item) {
        if (this.stack.length > 0) {
            const last = this.stack[this.stack.length - 1];

            const last_layer = last["layer"];
            const item_layer = item["layer"];

            if (last_layer == item_layer) {
                this.stack.pop();
                this._place(last);
            } else if (last_layer < item_layer) {
                this.stack.pop();
                this._place(last);
                this._build(item);
            }

        }
    }

    destroy() {
        this.nodes.destroy();
    }

}