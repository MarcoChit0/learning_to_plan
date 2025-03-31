(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	image1 - mode
	image0 - mode
	spectrograph2 - mode
	star0 - direction
	star1 - direction
	groundstation3 - direction
	groundstation4 - direction
	star2 - direction
	star5 - direction
	star6 - direction
	planet7 - direction
	star8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument2 spectrograph2)
	(supports instrument2 image1)
	(supports instrument2 image0)
	(calibration_target instrument2 star2)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star8)
)
(:goal (and
	(have_image star5 spectrograph2)
	(have_image star6 spectrograph2)
	(have_image planet7 image0)
	(have_image star8 image1)
	(have_image planet9 spectrograph2)
))

)
