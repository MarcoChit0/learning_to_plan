(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	image1 - mode
	infrared0 - mode
	star2 - direction
	star3 - direction
	star0 - direction
	star1 - direction
	planet4 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 image1)
	(calibration_target instrument0 star3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument1 infrared0)
	(supports instrument1 image1)
	(calibration_target instrument1 star2)
	(supports instrument2 image1)
	(calibration_target instrument2 star3)
	(supports instrument3 image1)
	(supports instrument3 infrared0)
	(calibration_target instrument3 star0)
	(supports instrument4 image1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(have_image planet4 infrared0)
))

)
