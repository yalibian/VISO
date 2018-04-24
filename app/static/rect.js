var width = 960,
    height = 200,
    divisions = 10;

var newData = [];
var sectionWidth = Math.floor(width / divisions);

for (var i=0; i < width; i+= sectionWidth ) {
    newData.push(i);
}

var colorScaleLin = d3.scale.linear()
      .domain([0, newData.length-1])
      .interpolate(d3.interpolateRgb)
      .range(['red', 'green']);

var colorScalePow = d3.scale.pow().exponent(.6)
      .domain([0, newData.length-1])
      .interpolate(d3.interpolateRgb)
      .range(['red', 'green']);

var vis = d3.select("#d3")
    .append("svg")
        .attr("width", width)
        .attr("height", height);

vis.selectAll('rect')
    .data(newData)
    .enter()
    .append('rect')
        .attr("x", function(d) { return d; })
        .attr("y", 0)
        .attr("height", height)
        .attr("width", sectionWidth)
.attr('fill', function(d, i) { return colorScaleLin(i)});