/**
 * CrowdSPIRE: Main function for Workspace
 * More function will be added sooner.
 */

const svg = d3.select('#workspace');
const WIDTH = parseInt(svg.style("width"), 10);
const HEIGHT = parseInt(svg.style("height"), 10);

const IconSide = 10;
const IconR = 3;


var nodes;


var forceCollide = d3.forceCollide()
    .radius(function (d) {
        return d.radius;
    })
    .iterations(2)
    .strength(0.95);

var simulation = d3.forceSimulation()
    .force("link", d3.forceLink().id(function (d) {
        return d.id;
    }))
    .force("charge", d3.forceManyBody())
    .force("center", d3.forceCenter(WIDTH / 2, HEIGHT / 2))
    .force("collide", forceCollide);


var labeledInstances = [];
var deletedInstances = [];
var TrainCount = 0;

var q = d3.queue();
q.defer(d3.csv, 'elements', elementType)
    .defer(d3.csv, 'links', linkType)
    .await(workspace);

function elementType(d, i) {
    d.id = '' + i;
    d.labeled = +d.labeled;
    d.prob = +d.prob;
    d.x = +d.prob * WIDTH;
    return d;
}

function linkType(d) {
    d.value = (+d.value);
    d.value = d.value * 0.0005;
    return d;
}


// Passing a bath of labeled instances back, training new classifier and update layout
function updateLayout() {

    if (labeledInstances.length <= 0) {
        // pop out window: no instances was labelled, please label some data through dragging
        return
    }

    d3.request('/training')
        .mimeType("text/csv")
        .post(JSON.stringify(labeledInstances), updateWorkspace);

    TrainCount++;
    labeledInstances = [];

}


// Cancel the passed bath of labeled instances, training old classifier and update layout
function undoLayout() {

    if (TrainCount <= 0) {
        return;
    }

    TrainCount--;
    // Asking the back end load old trained models
    d3.request('/undoTraining')
        .mimeType("text/csv")
        .get(updateWorkspace);
}


function resetLayout() {

    if (TrainCount <= 0) {
        return;
    }

    TrainCount = 0;
    // Asking the back end load old trained models
    d3.request('/resetTraining')
        .mimeType("text/csv")
        .get(updateWorkspace);
}


function undoInstance() {

    if (labeledInstances.length <= 0) {
        return;
    }

    // console.log('in undoInstance');
    simulation.alpha(0.3).restart();
    var ins = labeledInstances.pop();
    deletedInstances.push(JSON.parse(JSON.stringify(ins)));
    nodes.each(function (d) {

        if (('' + ins.id) === d.id) {
            d.fx = ins.orginialProb * WIDTH;
            d.fy = null;

            d3.select(this)
                .select('rect')
                .attr("fill", function (d) {
                    return '#efedf5';
                });
        }
    });

    // updateWorkspace();

}

function redoInstance() {

    if (deletedInstances.length <= 0) {
        return;
    }

    // simulation.alpha(0.3).restart();
    var ins = deletedInstances.pop();
    nodes.each(function (d) {

        if ('' + ins.id === d.id) {
            d.fx = ins.fx;
            d.prob = ins.prob;
            d.labeled = 1.0;

            d3.select(this)
                .select('rect')
                .attr("fill", function (d) {
                    if (d.labeled > 0.5) {
                        if (d.prob < 0.5) {
                            return '#2c7bb6';
                        } else {
                            return '#d7191c';
                        }
                    }
                    return '#efedf5';
                })
                .attr('fx', function (d) {

                    var prob = d.prob;
                    if (prob > 0.75 || prob < 0.25) {
                        d.fx = WIDTH * d.prob;
                        return d.fx;
                    } else {
                        d.fx = null;
                        return null;
                    }
                })
        }
    });

    labeledInstances.push(ins);
    simulation.alpha(0.3).restart();
}


function updateWorkspace() {

    simulation.alpha(0.3).restart();
    // simulation.alpha(0.3).restart();
    var q = d3.queue();
    q.defer(d3.csv, 'elements', elementType)
        .defer(d3.csv, 'links', linkType)
        .await(workspace);
}


// set the ranges
var x = d3.scaleTime().range([0, WIDTH]);
var y = d3.scaleLinear().range([HEIGHT, 0]);


// gridlines in x axis function
function make_x_gridlines() {
    return d3.axisBottom(x)
        .ticks(9)
}

// gridlines in y axis function
function make_y_gridlines() {
    return d3.axisLeft(y)
        .ticks(5)
}


// Main controller and view
// draw the workspace with docs
function workspace(error, elements, similarities) {

    // var edges;
    // if(similarities){
    //     edges = similarities_d;
    // }

    $('#workspace').empty();
    // var mainGradient = svg.append("defs")
        // .append('linearGradient')
        // .attr('id', 'mainGradient');

    // mainGradient.append('stop')
    // // .attr('class', 'stop-left')
    //     .attr('offset', '0%')
    //     .style('stop-color', '#2c7bb6')
    //     .style('stop-opacity', 0.45);
    //
    // mainGradient.append('stop')
    //     .attr('offset', '12.5%')
    //     .style('stop-color', '#00a6ca')
    //     .style('stop-opacity', 0.45);
    //
    // mainGradient.append('stop')
    //     .attr('offset', '25%')
    //     .style('stop-color', '#00ccbc')
    //     .style('stop-opacity', 0.45);
    //
    // mainGradient.append('stop')
    //     .attr('offset', '37.5%')
    //     .style('stop-color', '#90eb9d')
    //     .style('stop-opacity', 0.45);
    //
    // mainGradient.append('stop')
    //     .attr('offset', '50%')
    //     .style('stop-color', '#ffff8c')
    //     .style('stop-opacity', 0.45);
    //
    // mainGradient.append('stop')
    //     .attr('offset', '62.5%')
    //     .style('stop-color', '#f9d057')
    //     .style('stop-opacity', 0.45);
    //
    // mainGradient.append('stop')
    //     .attr('offset', '75%')
    //     .style('stop-color', '#f29e2e')
    //     .style('stop-opacity', 0.45);
    //
    // mainGradient.append('stop')
    //     .attr('offset', '87.5%')
    //     .style('stop-color', '#e76818')
    //     .style('stop-opacity', 0.45);
    //
    // mainGradient.append('stop')
    //     .attr('offset', '100%')
    //     .style('stop-color', '#d7191c')
    //     .style('stop-opacity', 0.45);


    // add the X gridlines
    svg.append("g")
        .attr("class", "grid")
        // .attr("transform", "translate(0," + height + ")")
        .call(make_x_gridlines()
            .tickSize(WIDTH)
            .tickFormat("")
        );

    // add the Y gridlines
    svg.append("g")
        .attr("class", "grid")
        .call(make_y_gridlines()
            .tickSize(WIDTH / 10)
            .tickFormat("")
        );


    // d3.select("#workspace").empty();
    // svg.append('rect')
    //     .attrs({x: 0, y: 0, width: WIDTH, height: HEIGHT, fill: "url(#mainGradient)"});


    if (error) {
        throw error;
    }

    var edges = similarities;

    simulation.nodes(elements)
        .on("tick", ticked);

    simulation.force("link")
        .links(similarities)
        .strength(function (link) {
            return link.value;
        });


    nodes = svg.selectAll(".node")
        .data(elements)
        .enter()
        .append("g")
        .attr("class", "node")
        .on("mousedown", function () {
            d3.event.preventDefault();
        })
        .call(d3.drag()
            .on("start", nodeDragStarted)
            .on("drag", nodeDragged)
            .on("end", nodeDragEnded));

    // label
    nodes.append("text")
        .attr("dx", 12)
        .attr("dy", ".35em")
        .text(function (d) {
            return d.Animal;
        });

    // rectangle
    nodes.append("rect")
        .attr("width", function (d) {
            d.radius = IconSide / Math.sqrt(2.00);
            d.width = IconSide;
            return IconSide;
        })
        .attr("height", function (d) {
            d.height = IconSide;
            return IconSide;
        })
        .attr("x", function (d) {
            return d.width;
        })
        .attr("y", function (d) {
            return d.height;
        })
        .attr("fill", function (d) {
            if (d.labeled > 0.5) {

                // console.log(d);

                if (d.prob < 0.5) {
                    return '#2c7bb6';
                } else {
                    return '#d7191c';
                }
            }
            return '#efedf5';
        })
        .attr('fx', function (d) {

            if (d.labeled > 0.2) {
                // console.log(+d.prob);

                d.fx = WIDTH * d.prob;
                return WIDTH * d.prob;
            }

            var prob = d.prob;
            if (prob !== 0.5) {

                d.fx = WIDTH * d.prob;
                return d.fx;
            }

            d.fx = null;
            return null;
        })
        .attr('rx', function (d) {

            return IconR;
        })
        .attr('ry', function (d) {

            return IconR;
        })
        .attr('class', 'IconRect');

    // ticked
    function ticked() {

        // Not change the width and height of each node.
        nodes.attr("transform", function (d) {

            // border constriction
            // width = 20
            d.x = Math.max(60, Math.min(WIDTH - 60, d.x));
            d.y = Math.max(d.height / 2, Math.min(HEIGHT - d.height / 2, d.y));

            // Update the group position: which include the basic rectangle and Foreign object. (Drag Object too...)
            return "translate(" + d.x + "," + d.y + ")";
        });

        // rectangle: keep ICON rectangle at the center of Node Group
        nodes.selectAll("rect")
            .attr("x", function (d) {
                return -d.width / 2;
            })
            .attr("y", function (d) {
                return -d.height / 2;
            });

    }


    function nodeDragStarted(d) {
        d3.select(this).moveToFront();
    }

    // During node drag.
    function nodeDragged(d) {
        d.fx = d3.event.x;
        d.fy = d3.event.y;
    }

    // the end of node drag.
    function nodeDragEnded(d) {

        // Feedback to server the labeled data
        var labeledInstance = {};
        // labeledInstance.id = d.id;
        labeledInstance.id = +d.id;
        labeledInstance.name = d.Animal;
        labeledInstance.orginialProb = d.prob;

        // console.log(d.x/WIDTH);
        labeledInstance.prob = d3.event.x / WIDTH;


        labeledInstances.push(labeledInstance);

        console.log(labeledInstance);

        d3.select(this)
            .select('rect')
            .attr("fill", function (d) {
                if (labeledInstance.prob < 0.5) {
                    return '#2c7bb6';
                } else {
                    return '#d7191c';
                }
            });

        if (!d3.event.active) {
            simulation.alphaTarget(0);
        }
        forceCollide.initialize(simulation.nodes());

        // Update and restart the simulation.
        simulation.nodes(elements);
        simulation.force("link").links(edges);
        simulation.alpha(0.3).restart();
    }

}

