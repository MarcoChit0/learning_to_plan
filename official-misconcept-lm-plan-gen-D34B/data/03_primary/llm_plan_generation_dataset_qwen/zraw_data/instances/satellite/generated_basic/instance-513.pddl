(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	spectrograph0 - mode
	image2 - mode
	spectrograph3 - mode
	infrared1 - mode
	star1 - direction
	star4 - direction
	star2 - direction
	groundstation3 - direction
	groundstation0 - direction
	star5 - direction
	star6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(supports instrument0 image2)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star2)
	(supports instrument1 image2)
	(supports instrument1 spectrograph0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument2 infrared1)
	(supports instrument2 image2)
	(supports instrument2 spectrograph3)
	(calibration_target instrument2 groundstation0)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
)
(:goal (and
	(have_image star5 image2)
	(have_image star6 spectrograph0)
	(have_image planet7 image2)
	(have_image planet8 spectrograph0)
))

)
