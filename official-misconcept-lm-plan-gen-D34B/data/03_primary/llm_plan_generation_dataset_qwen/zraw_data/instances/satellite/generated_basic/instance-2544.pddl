(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	infrared1 - mode
	spectrograph0 - mode
	image2 - mode
	groundstation0 - direction
	star2 - direction
	star4 - direction
	groundstation1 - direction
	groundstation3 - direction
	star5 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation3)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
)
(:goal (and
	(pointing satellite0 star5)
	(have_image star5 image2)
))

)
