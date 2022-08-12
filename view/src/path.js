class Node {
    constructor(parent, property, col, row) {

        this.parts = new Konva.Group();

        const rad = 12;
        const x = col * 84;
        const y = row * 36;

        const icon = new Konva.Circle({
            x: x,
            y: y,
            radius: rad,
            fill: "#fff",
            stroke: "black",
            strokeWidth: 2,
            shadowBlur: 2,
            shadowOffset: { x: 2, y: 2 },
            shadowOpacity: 0.9,
        });
        this.icon = icon;

        const text = new Konva.Text({
            x: x - rad,
            y: y - rad,
            text: "" + property["selected"],
            fontSize: 18,
            fontFamily: "Calibri",
            fill: "#000",
            width: rad * 2,
            height: rad * 2,
            padding: 0,
            align: "center",
            verticalAlign: "middle",
            visible: true,
        });

        this.parts.add(icon);
        this.parts.add(text);

        this.info = new Konva.Group();

        let offset_x = x;
        let offset_y = y + icon.height();
        for (const choice of property["choices"]) {

            const icon = new Konva.Rect({
                x: offset_x,
                y: offset_y,
                width: 60,
                height: rad * 2,
                fill: (property["selected"] === choice[0] ? "rgba(0, 255, 255, 1.0)" : "#fff"),
                stroke: "black",
                strokeWidth: 2,
                shadowBlur: 2,
                shadowOffset: { x: 2, y: 2 },
                shadowOpacity: 0.9,
            });

            const text = new Konva.Text({
                x: offset_x,
                y: offset_y,
                text: "" + choice[0] + ":" + choice[1].toFixed(2),
                fontSize: 18,
                fontFamily: "Calibri",
                fill: "#000",
                width: icon.width(),
                height: icon.height(),
                padding: 4,
                align: "left",
                verticalAlign: "middle",
                visible: true,
            });

            offset_y = offset_y + icon.height();

            this.info.add(icon);
            this.info.add(text);

        }

        parent.add(this.parts);
        parent.add(this.info);

        this.start_point = {"x": this.icon.x() - rad, "y": this.icon.y()};
        this.end_point = {"x": this.icon.x() + rad, "y": this.icon.y()};
        this.layer = property["layer"];
    }

    destroy() {
        this.parts.destroy();
        this.info.destroy();
    }
}


class Path_hierarchy {

    constructor(parent_ui, parent_bg, path_data, handle_selection) {
        console.log(path_data);
        this.handle_selection = handle_selection;

        this.nodes = [];
        this.arrow_group = new Konva.Group();
        this.draw_group = new Konva.Group();

        this.step = 0;
        this.layers = {};

        for (const item of path_data) {
            if (item["s"] != null) {
                this.layers[item["layer"]] = item;
                continue;
            }
        }

        for (const item of path_data) {
            if (item["s"] != null) {
                continue;
            }
            this._place(item);
        }

        parent_bg.add(this.arrow_group);
        parent_bg.add(this.draw_group);
    }

    _place(item) {
        const item_layer = item["layer"];
        const node = new Node(this.draw_group, item, this.step++, (Object.keys(this.layers).length - 1) - item_layer);
        
        this._link(node);
        this.nodes.push(node);
    }


    _link(node) {
        
        let start = null;
        for(let i = this.nodes.length - 1;i>=0;--i) {
            if(this.nodes[i].layer === node.layer) {
                start = this.nodes[i].end_point;
                break;
            }
        }
        if(start == null) return;
        
        const end = node.start_point;

        const arrow = new Konva.Arrow({
            points: [start["x"], start["y"], end["x"], end["y"]],
            pointerLength: 6,
            pointerWidth: 6,
            stroke: 'black',
            strokeWidth: 4,
        });
        this.arrow_group.add(arrow);
    }

    destroy() {
        this.arrow_group.destroy();
        this.draw_group.destroy();
        this.nodes = [];
    }

}