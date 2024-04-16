window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked_new";
var NUM_INTERP_FRAMES = 104;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i <= NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/interpolate (' + String(i) + ').jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}

var EXAMPLE1_BASE = "./static/interpolation/example1";
var NUM_INTERP_FRAMES = 20;

var example1_images = [];
function preloadExample1Images() {
  for (var i = 0; i <= NUM_INTERP_FRAMES; i++) {
    var path = EXAMPLE1_BASE + '/visualization (' + String(i) + ').jpg';
    example1_images[i] = new Image();
    example1_images[i].src = path;
  }
}

function setExample1Image(i) {
  var image = example1_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}



$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES);

    $('#example1-slider').on('input', function(event) {
      setExample1Image(this.value);
    });
    setExample1Image(0);
    $('#interpolation-slider').prop('max', NUM_EXAMPLE1_FRAMES);

    bulmaSlider.attach();

})
