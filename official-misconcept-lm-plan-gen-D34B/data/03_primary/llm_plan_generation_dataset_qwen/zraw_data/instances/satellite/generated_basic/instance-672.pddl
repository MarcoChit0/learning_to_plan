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
	star0 - direction
	star2 - direction
	star3 - direction
	groundstation1 - direction
	star5 - direction
	groundstation4 - direction
	planet6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 star3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation4)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 image1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 star5)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
)
(:goal (and
	(have_image planet6 infrared0)
	(have_image planet7 image1)
	(have_image planet8 infrared0)
))

)
