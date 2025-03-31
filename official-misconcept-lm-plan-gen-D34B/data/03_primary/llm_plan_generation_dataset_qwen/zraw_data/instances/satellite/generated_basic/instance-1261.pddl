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
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	thermograph1 - mode
	image0 - mode
	star0 - direction
	planet1 - direction
	planet2 - direction
	star3 - direction
	planet4 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet2)
	(supports instrument1 image0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph1)
	(supports instrument2 image0)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 star0)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(supports instrument5 image0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet1)
	(supports instrument6 image0)
	(calibration_target instrument6 star0)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet1)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star0)
	(supports instrument8 image0)
	(calibration_target instrument8 star0)
	(supports instrument9 image0)
	(calibration_target instrument9 star0)
	(supports instrument10 image0)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 star0)
	(supports instrument11 image0)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 star0)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet1)
	(supports instrument12 image0)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 star0)
	(supports instrument13 image0)
	(calibration_target instrument13 star0)
	(supports instrument14 image0)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 star0)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet4)
)
(:goal (and
	(pointing satellite0 star3)
	(pointing satellite2 star0)
	(pointing satellite4 planet2)
	(have_image planet1 image0)
	(have_image planet2 image0)
	(have_image star3 image0)
	(have_image planet4 image0)
))

)
