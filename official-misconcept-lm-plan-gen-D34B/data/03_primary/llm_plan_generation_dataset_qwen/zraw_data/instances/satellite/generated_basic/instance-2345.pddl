(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	image1 - mode
	infrared0 - mode
	infrared2 - mode
	star2 - direction
	star3 - direction
	star4 - direction
	groundstation1 - direction
	star0 - direction
	planet5 - direction
	star6 - direction
	phenomenon7 - direction
	planet8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared2)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument2 infrared2)
	(supports instrument2 infrared0)
	(supports instrument2 image1)
	(calibration_target instrument2 star0)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet8)
)
(:goal (and
	(pointing satellite1 star6)
	(have_image planet5 infrared2)
	(have_image star6 image1)
	(have_image phenomenon7 infrared2)
	(have_image planet8 image1)
	(have_image phenomenon9 image1)
))

)
