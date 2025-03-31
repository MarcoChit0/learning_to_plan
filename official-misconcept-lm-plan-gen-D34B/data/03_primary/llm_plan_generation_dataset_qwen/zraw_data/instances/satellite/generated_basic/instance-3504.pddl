(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	thermograph2 - mode
	image1 - mode
	image0 - mode
	star1 - direction
	star0 - direction
	star2 - direction
	star3 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star1)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument2 image1)
	(calibration_target instrument2 star1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument3 image0)
	(calibration_target instrument3 star1)
	(supports instrument4 thermograph2)
	(supports instrument4 image1)
	(supports instrument4 image0)
	(calibration_target instrument4 star1)
	(supports instrument5 image0)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 star0)
	(supports instrument6 thermograph2)
	(supports instrument6 image1)
	(supports instrument6 image0)
	(calibration_target instrument6 star1)
	(supports instrument7 image0)
	(supports instrument7 image1)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 star0)
	(supports instrument8 thermograph2)
	(supports instrument8 image0)
	(supports instrument8 image1)
	(calibration_target instrument8 star0)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument9 image0)
	(calibration_target instrument9 star0)
	(supports instrument10 thermograph2)
	(calibration_target instrument10 star0)
	(supports instrument11 image0)
	(supports instrument11 thermograph2)
	(calibration_target instrument11 star0)
	(supports instrument12 image0)
	(supports instrument12 image1)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 star1)
	(supports instrument13 thermograph2)
	(supports instrument13 image0)
	(calibration_target instrument13 star0)
	(supports instrument14 image1)
	(calibration_target instrument14 star0)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
)
(:goal (and
	(pointing satellite1 star2)
	(pointing satellite3 star1)
	(have_image star2 thermograph2)
	(have_image star3 image0)
))

)
