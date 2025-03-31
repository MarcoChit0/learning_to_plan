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
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	infrared0 - mode
	thermograph2 - mode
	thermograph3 - mode
	image1 - mode
	star0 - direction
	star2 - direction
	star1 - direction
	star3 - direction
	star4 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 image1)
	(calibration_target instrument0 star3)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 star2)
	(supports instrument2 thermograph2)
	(supports instrument2 image1)
	(calibration_target instrument2 star3)
	(supports instrument3 thermograph3)
	(supports instrument3 thermograph2)
	(supports instrument3 image1)
	(calibration_target instrument3 star2)
	(supports instrument4 thermograph3)
	(calibration_target instrument4 star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument5 infrared0)
	(supports instrument5 image1)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 star1)
	(supports instrument6 thermograph3)
	(supports instrument6 thermograph2)
	(supports instrument6 image1)
	(calibration_target instrument6 star1)
	(supports instrument7 image1)
	(calibration_target instrument7 star1)
	(supports instrument8 thermograph3)
	(calibration_target instrument8 star3)
	(supports instrument9 thermograph3)
	(supports instrument9 image1)
	(calibration_target instrument9 star3)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
)
(:goal (and
	(pointing satellite1 star0)
	(have_image star4 thermograph3)
))

)
