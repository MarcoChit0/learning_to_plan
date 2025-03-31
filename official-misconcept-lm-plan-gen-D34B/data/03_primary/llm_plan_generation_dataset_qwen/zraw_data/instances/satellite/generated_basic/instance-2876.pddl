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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	thermograph2 - mode
	thermograph3 - mode
	infrared0 - mode
	image1 - mode
	star2 - direction
	star1 - direction
	star3 - direction
	star0 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	star6 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 thermograph3)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star3)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph2)
	(supports instrument2 image1)
	(calibration_target instrument2 star1)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star1)
	(supports instrument4 image1)
	(supports instrument4 infrared0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument5 thermograph3)
	(supports instrument5 image1)
	(calibration_target instrument5 star2)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument6 image1)
	(supports instrument6 infrared0)
	(supports instrument6 thermograph3)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared0)
	(calibration_target instrument7 star1)
	(supports instrument8 infrared0)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 star0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument9 thermograph3)
	(calibration_target instrument9 star2)
	(supports instrument10 image1)
	(calibration_target instrument10 star0)
	(supports instrument11 thermograph3)
	(supports instrument11 infrared0)
	(supports instrument11 thermograph2)
	(calibration_target instrument11 star1)
	(supports instrument12 thermograph3)
	(supports instrument12 image1)
	(calibration_target instrument12 star3)
	(supports instrument13 thermograph3)
	(supports instrument13 image1)
	(supports instrument13 infrared0)
	(calibration_target instrument13 star3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star6)
	(supports instrument14 image1)
	(supports instrument14 thermograph3)
	(calibration_target instrument14 star0)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
)
(:goal (and
	(pointing satellite3 phenomenon5)
	(have_image phenomenon4 infrared0)
	(have_image phenomenon5 infrared0)
	(have_image star6 infrared0)
))

)
