(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	image1 - mode
	infrared0 - mode
	infrared2 - mode
	star0 - direction
	star1 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 infrared0)
	(supports instrument0 image1)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared0)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared0)
	(supports instrument2 image1)
	(supports instrument2 infrared2)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 infrared0)
	(supports instrument3 infrared2)
	(supports instrument3 image1)
	(calibration_target instrument3 star0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite1 star1)
	(have_image star1 infrared2)
))

)
