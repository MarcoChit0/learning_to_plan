(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	infrared0 - mode
	image1 - mode
	groundstation2 - direction
	star0 - direction
	star3 - direction
	star1 - direction
	star4 - direction
	star5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star3)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(supports instrument2 image1)
	(supports instrument2 infrared0)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared0)
	(supports instrument3 image1)
	(calibration_target instrument3 star1)
	(supports instrument4 infrared0)
	(supports instrument4 image1)
	(calibration_target instrument4 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument5 infrared0)
	(supports instrument5 image1)
	(calibration_target instrument5 star1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
)
(:goal (and
	(pointing satellite0 star5)
	(have_image star4 infrared0)
	(have_image star5 infrared0)
	(have_image planet6 infrared0)
))

)
