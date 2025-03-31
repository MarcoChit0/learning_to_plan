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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	image1 - mode
	infrared0 - mode
	image2 - mode
	thermograph3 - mode
	star0 - direction
	star1 - direction
	star2 - direction
	star3 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 thermograph3)
	(supports instrument0 image2)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph3)
	(supports instrument1 image2)
	(calibration_target instrument1 star0)
	(supports instrument2 image1)
	(supports instrument2 infrared0)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 thermograph3)
	(supports instrument3 image1)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared0)
	(supports instrument4 thermograph3)
	(calibration_target instrument4 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument5 thermograph3)
	(supports instrument5 image1)
	(supports instrument5 image2)
	(calibration_target instrument5 star0)
	(supports instrument6 infrared0)
	(calibration_target instrument6 star0)
	(supports instrument7 image2)
	(calibration_target instrument7 star0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(have_image star1 image2)
	(have_image star2 image2)
	(have_image star3 image1)
))

)
