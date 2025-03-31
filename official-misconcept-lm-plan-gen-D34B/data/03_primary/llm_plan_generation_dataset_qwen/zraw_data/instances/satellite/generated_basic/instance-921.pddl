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
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	instrument15 - instrument
	satellite5 - satellite
	instrument16 - instrument
	instrument17 - instrument
	image0 - mode
	thermograph1 - mode
	star1 - direction
	star0 - direction
	star2 - direction
	groundstation3 - direction
	planet4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument1 thermograph1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 image0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star2)
	(supports instrument3 image0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph1)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(supports instrument5 image0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star1)
	(supports instrument6 image0)
	(calibration_target instrument6 star0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet4)
	(supports instrument7 image0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star1)
	(supports instrument8 image0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star0)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 star0)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 star1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 groundstation3)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 star1)
	(supports instrument13 thermograph1)
	(supports instrument13 image0)
	(calibration_target instrument13 star2)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
	(supports instrument14 image0)
	(calibration_target instrument14 star0)
	(supports instrument15 thermograph1)
	(supports instrument15 image0)
	(calibration_target instrument15 star2)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
	(supports instrument16 thermograph1)
	(supports instrument16 image0)
	(calibration_target instrument16 groundstation3)
	(supports instrument17 image0)
	(supports instrument17 thermograph1)
	(calibration_target instrument17 groundstation3)
	(on_board instrument16 satellite5)
	(on_board instrument17 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star0)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite2 star0)
	(pointing satellite5 star2)
	(have_image planet4 image0)
	(have_image phenomenon5 image0)
))

)
