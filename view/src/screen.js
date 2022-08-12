screen = function() {

    const gen_button = function(logo) {
        const button_group = new Konva.Group();

        const bg = new Konva.Rect({
            x: 0,
            y: 0,
            width: 1.0,
            height: 1.0,
            offsetX: 0.5,
            offsetY: 0.5,
            fill: 'white',
            strokeWidth: 0.04,
            stroke: 'black',
            shadowBlur: 0.05,
            cornerRadius: 0.1,
        });

        logo.scale({ x: 1.0 / logo.width(), y: 1.0 / logo.height() });
        button_group.add(bg);
        button_group.add(logo);
        return button_group;
    }

    const close_svg = new Konva.Path({
        x: 0,
        y: 0,
        width: 24,
        height: 24,
        offsetX: 12,
        offsetY: 12,
        data: 'M12 1C5.926 1 1 5.926 1 12s4.926 11 11 11 11-4.926 11-11S18.074 1 12 1zm5.25 14.75l-1.5 1.5L12 13.5l-3.75 3.75-1.5-1.5L10.5 12 6.75 8.25l1.5-1.5L12 10.5l3.75-3.75 1.5 1.5L13.5 12l3.75 3.75z',
        fill: 'red',
    });

    const button_close = gen_button(close_svg);

    const next_svg = new Konva.Path({
        x: 0,
        y: 0,
        width: 16,
        height: 16,
        offsetX: 8,
        offsetY: 8,
        data: 'M2 2.96495C2 2.15413 2.91427 1.68039 3.57668 2.14798L10.7097 7.18302C11.2741 7.58143 11.2741 8.41854 10.7097 8.81695L3.57668 13.852C2.91427 14.3196 2 13.8458 2 13.035V2.96495zM14 2.75C14 2.33579 13.6642 2 13.25 2 12.8358 2 12.5 2.33579 12.5 2.75V13.25C12.5 13.6642 12.8358 14 13.25 14 13.6642 14 14 13.6642 14 13.25V2.75z',
        fill: 'black',
    });

    const button_next = gen_button(next_svg);
    const button_prev = button_next.clone();

    const undo_svg = new Konva.Path({
        x: 0,
        y: 0,
        width: 230,
        height: 230,
        offsetX: 90.6665,
        offsetY: 90.6665,
        data: 'M125.578,181.333C145.718,144.845,149.112,89.193,70,91.052V136L2,68,70,0V43.985C164.735,41.514,175.286,127.607,125.578,181.333Z',
        fill: 'black',
    });

    const button_undo = gen_button(undo_svg);
    const button_redo = button_undo.clone();

    const button_zoom_in = gen_button(new Konva.Path({
        x: 0,
        y: 0,
        width: 24,
        height: 24,
        offsetX: 12,
        offsetY: 12,
        data: 'M20.71 19.29l-3.4-3.39A7.92 7.92 0 0 0 19 11a8 8 0 1 0-8 8 7.92 7.92 0 0 0 4.9-1.69l3.39 3.4a1 1 0 0 0 1.42 0 1 1 0 0 0 0-1.42zM13 12h-1v1a1 1 0 0 1-2 0v-1H9a1 1 0 0 1 0-2h1V9a1 1 0 0 1 2 0v1h1a1 1 0 0 1 0 2z',
        fill: 'black',
    }));

    const button_zoom_out = gen_button(new Konva.Path({
        x: 0,
        y: 0,
        width: 24,
        height: 24,
        offsetX: 12,
        offsetY: 12,
        data: 'M20.71 19.29l-3.4-3.39A7.92 7.92 0 0 0 19 11a8 8 0 1 0-8 8 7.92 7.92 0 0 0 4.9-1.69l3.39 3.4a1 1 0 0 0 1.42 0 1 1 0 0 0 0-1.42zM13 12H9a1 1 0 0 1 0-2h4a1 1 0 0 1 0 2z',
        fill: 'black',
    }));


    let screen_obj = {};
    let stage;
    let start_width;

    function fitStageIntoParentContainer() {
        const container = document.querySelector('#stage-parent');

        // now we need to fit stage into parent container
        const containerWidth = container.offsetWidth;
        const containerHeight = container.offsetHeight;

        stage.width(containerWidth);
        stage.height(containerHeight);

        const button_size = Math.min(stage.width(), stage.height()) * 0.1;
        const button_y = stage.height() * 0.90;

        button_close.position({ x: stage.width() * 0.50, y: button_y });
        button_close.scale({ x: button_size, y: button_size });

        button_prev.position({ x: stage.width() * 0.10, y: button_y });
        button_prev.scale({ x: -button_size, y: button_size });

        button_undo.position({ x: stage.width() * 0.20, y: button_y });
        button_undo.scale({ x: button_size, y: button_size });

        button_redo.position({ x: stage.width() * 0.30, y: button_y });
        button_redo.scale({ x: -button_size, y: button_size });

        button_zoom_in.position({ x: stage.width() * 0.70, y: button_y });
        button_zoom_in.scale({ x: button_size, y: button_size });

        button_zoom_out.position({ x: stage.width() * 0.80, y: button_y });
        button_zoom_out.scale({ x: button_size, y: button_size });

        button_next.position({ x: stage.width() * 0.90, y: button_y });
        button_next.scale({ x: button_size, y: button_size });
    }

    const is_touch_device = (('ontouchstart' in window) ||
        (navigator.maxTouchPoints > 0) ||
        (navigator.msMaxTouchPoints > 0));

    screen_obj.on_load = function(data) {

        let container = document.querySelector('#stage-parent');
        let startWidth = container.offsetWidth;
        let startHeight = container.offsetHeight;

        stage = new Konva.Stage({
            container: 'container',
            // first just set set as is
            width: startWidth,
            height: startHeight,
        });
        window.stage = stage; //for text nodes access.

        start_width = startWidth;

        stage.container().style.backgroundColor = '#EEE';

        fitStageIntoParentContainer();

        // then create layers
        const canvas_layer = new Konva.Layer();
        const ui_layer = new Konva.Layer();

        const work_group = new Konva.Group({
            x: stage.width()/2,
            y: stage.height()/2,
        });
        work_group.draggable(!is_touch_device);

        const work_group_ui = new Konva.Group();
        const work_group_bg = new Konva.Group();
        work_group.add(work_group_bg);
        work_group.add(work_group_ui);

        function handle_update(object, type) {
        }

        let working_path = null;

        function new_path(path_object) {
            working_path = new Path_hierarchy(work_group_ui, work_group_bg, path_object, handle_update);
        }

        let current_path = -1;

        function next_path() {
            if(working_path != null) working_path.destroy();
            current_path = current_path + 1;
            if (current_path > data.length - 1) current_path = data.length - 1;
            new_path(data[current_path]);
        }

        function prev_path() {
            if(working_path != null) working_path.destroy();
            current_path = current_path - 1;
            if (current_path < 0) current_path = 0;
            new_path(data[current_path]);
        }

        let lastDist = 0;
        let lastCenter = null;

        work_group.on('mouseup touchend', function(e) {
            if (window.touch_bubble_flag) {
                window.touch_bubble_flag = false;
                return
            }
            if (lastCenter != null) {
                setTimeout(function() {
                    lastDist = 0;
                    lastCenter = null;
                    work_group.draggable(!is_touch_device);
                }, 30);
            } else {
                let pos = work_group.getRelativePointerPosition();
            }
        });

        work_group.on('pointermove', function(e) {
            if (lastCenter != null) {} else {
                let pos = work_group.getRelativePointerPosition();
            }
        });


        function zoom(pivot, dscale) {
            let scale = work_group.scale().x;
            let tl = work_group.position();
            work_group.scale({ x: scale * dscale, y: scale * dscale });
            work_group.position({ x: (tl.x - pivot.x) * dscale + pivot.x, y: (tl.y - pivot.y) * dscale + pivot.y });
        }

        work_group.on('wheel', function(e) {
            let pos = canvas_layer.getRelativePointerPosition();
            if (e.evt.deltaY > 0) {
                dscale = 0.95;
            } else if (e.evt.deltaY < 0) {
                dscale = 1.05;
            }
            zoom(pos, dscale);
        });

        work_group.on('touchmove', function(e) {
            e.evt.preventDefault();
            if (e.evt.touches.length >= 2) {
                let touch1 = e.evt.touches[0];
                let touch2 = e.evt.touches[1];

                work_group.draggable(false);

                if (touch1 && touch2) {

                    let p1 = {
                        x: touch1.clientX,
                        y: touch1.clientY,
                    };
                    let p2 = {
                        x: touch2.clientX,
                        y: touch2.clientY,
                    };

                    if (!lastCenter) {
                        lastCenter = util.center(p1, p2);
                        return;
                    }
                    let newCenter = util.center(p1, p2);

                    let dist = util.dist(p1, p2);

                    if (!lastDist) {
                        lastDist = dist;
                    }

                    // local coordinates of center point
                    let pointTo = {
                        x: (newCenter.x - work_group.x()) / work_group.scaleX(),
                        y: (newCenter.y - work_group.y()) / work_group.scaleX(),
                    };

                    let scale = work_group.scaleX() * (dist / lastDist);

                    work_group.scaleX(scale);
                    work_group.scaleY(scale);

                    // calculate new position of the stage
                    let dx = newCenter.x - lastCenter.x;
                    let dy = newCenter.y - lastCenter.y;

                    let newPos = {
                        x: newCenter.x - pointTo.x * scale + dx,
                        y: newCenter.y - pointTo.y * scale + dy,
                    };

                    work_group.position(newPos);

                    lastDist = dist;
                    lastCenter = newCenter;
                }
            }

        });

        canvas_layer.add(work_group);
        stage.add(canvas_layer);

        button_close.on('mouseup touchend', function(e) {
            e.evt.preventDefault();
        });


        button_next.on('mouseup touchend', function(e) {
            e.evt.preventDefault();
            next_path();
        });

        button_prev.on('mouseup touchend', function(e) {
            e.evt.preventDefault();
            prev_path();
        });

        button_undo.on('mouseup touchend', function(e) {
            e.evt.preventDefault();
        });

        button_redo.on('mouseup touchend', function(e) {
            e.evt.preventDefault();
        });

        button_zoom_in.on('mouseup touchend', function(e) {
            e.evt.preventDefault();
            zoom({ x: stage.width() / 2, y: stage.height() / 2 }, 1.05);
        });

        button_zoom_out.on('mouseup touchend', function(e) {
            e.evt.preventDefault();
            zoom({ x: stage.width() / 2, y: stage.height() / 2 }, 0.95);
        });

        ui_layer.add(button_close);
        ui_layer.add(button_next);
        ui_layer.add(button_prev);
        ui_layer.add(button_undo);
        ui_layer.add(button_redo);
        ui_layer.add(button_zoom_in);
        ui_layer.add(button_zoom_out);
        stage.add(ui_layer);


        // make it focusable
        stage.container().tabIndex = 1;
        stage.container().focus();

        stage.container().addEventListener('keydown', function(e) {
            if (event.getModifierState("Control")) {
                if (e.keyCode === 90) {
                    console.log("undo !");
                    e.preventDefault();
                } else if (e.keyCode === 89) {
                    console.log("redo !");
                    e.preventDefault();
                }
            }
        });

        return new Promise(function(resolve, reject) {
            next_path();
            resolve();
        });
    }

    // adapt the stage on any window resize
    window.addEventListener('resize', fitStageIntoParentContainer);

    return screen_obj;
}();