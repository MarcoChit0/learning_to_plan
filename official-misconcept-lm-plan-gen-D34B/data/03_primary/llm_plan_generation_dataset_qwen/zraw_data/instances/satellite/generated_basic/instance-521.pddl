(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	infrared2 - mode
	infrared0 - mode
	infrared3 - mode
	image1 - mode
	star2 - direction
	star1 - direction
	star4 - direction
	star3 - direction
	star0 - direction
	star5 - direction
	phenomenon6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared3)
	(calibration_target instrument0 star1)
	(supports instrument1 infrared0)
	(supports instrument1 infrared3)
	(supports instrument1 image1)
	(calibration_target instrument1 star4)
	(supports instrument2 image1)
	(supports instrument2 infrared2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet8)
	(supports instrument3 infrared0)
	(supports instrument3 infrared2)
	(supports instrument3 image1)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared2)
	(supports instrument4 image1)
	(supports instrument4 infrared3)
	(calibration_target instrument4 star0)
	(supports instrument5 image1)
	(calibration_target instrument5 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
)
(:goal (and
	(have_image star5 image1)
	(have_image phenomenon6 infrared2)
	(have_image planet7 infrared2)
	(have_image planet8 infrared2)
))

)
