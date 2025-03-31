(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	infrared0 - mode
	spectrograph1 - mode
	infrared2 - mode
	star0 - direction
	star4 - direction
	star2 - direction
	groundstation3 - direction
	star1 - direction
	planet5 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star2)
	(supports instrument1 infrared2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(pointing satellite0 planet5)
	(have_image planet5 infrared0)
))

)
