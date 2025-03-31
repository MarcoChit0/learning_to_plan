(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	infrared0 - mode
	star2 - direction
	star3 - direction
	star4 - direction
	star5 - direction
	star0 - direction
	groundstation1 - direction
	phenomenon6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star5)
	(calibration_target instrument0 star4)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation1)
	(calibration_target instrument2 star0)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
)
(:goal (and
	(have_image phenomenon6 infrared0)
	(have_image planet7 infrared0)
	(have_image planet8 infrared0)
))

)
