(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	image3 - mode
	spectrograph1 - mode
	image2 - mode
	spectrograph0 - mode
	star0 - direction
	groundstation2 - direction
	groundstation3 - direction
	star1 - direction
	star4 - direction
	planet5 - direction
	planet6 - direction
	star7 - direction
	star8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 image2)
	(supports instrument1 image3)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet9)
	(supports instrument2 image2)
	(calibration_target instrument2 star1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
)
(:goal (and
	(have_image star4 image3)
	(have_image planet5 spectrograph1)
	(have_image planet6 spectrograph1)
	(have_image star7 image3)
	(have_image star8 image3)
	(have_image planet9 image3)
))

)
