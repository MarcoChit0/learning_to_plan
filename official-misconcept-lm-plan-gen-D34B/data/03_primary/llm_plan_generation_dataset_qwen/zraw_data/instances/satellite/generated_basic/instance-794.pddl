(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	spectrograph0 - mode
	image1 - mode
	groundstation0 - direction
	star4 - direction
	star2 - direction
	star3 - direction
	groundstation1 - direction
	planet5 - direction
	phenomenon6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 star2)
	(supports instrument1 image1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 spectrograph0)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
)
(:goal (and
	(pointing satellite0 star2)
	(have_image planet5 image1)
	(have_image phenomenon6 spectrograph0)
	(have_image planet7 image1)
	(have_image phenomenon8 spectrograph0)
))

)
