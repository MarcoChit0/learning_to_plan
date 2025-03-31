(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	infrared1 - mode
	infrared2 - mode
	image0 - mode
	groundstation1 - direction
	star2 - direction
	star4 - direction
	star0 - direction
	star3 - direction
	planet5 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 star3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument1 infrared2)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared2)
	(supports instrument2 image0)
	(calibration_target instrument2 star3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
)
(:goal (and
	(have_image planet5 infrared2)
))

)
