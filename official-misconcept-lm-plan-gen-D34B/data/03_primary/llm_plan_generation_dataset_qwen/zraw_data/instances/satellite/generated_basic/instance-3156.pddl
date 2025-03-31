(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	satellite4 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	satellite5 - satellite
	instrument17 - instrument
	image0 - mode
	thermograph1 - mode
	star0 - direction
	groundstation1 - direction
	star2 - direction
	planet3 - direction
	phenomenon4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 thermograph1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet3)
	(supports instrument2 image0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star0)
	(supports instrument3 image0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star0)
	(supports instrument4 image0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet3)
	(supports instrument5 image0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 star0)
	(supports instrument6 image0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star2)
	(supports instrument7 image0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star2)
	(supports instrument8 image0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 thermograph1)
	(supports instrument9 image0)
	(calibration_target instrument9 star2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet5)
	(supports instrument10 image0)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 star0)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument11 thermograph1)
	(supports instrument11 image0)
	(calibration_target instrument11 star0)
	(supports instrument12 image0)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 star2)
	(supports instrument13 thermograph1)
	(supports instrument13 image0)
	(calibration_target instrument13 star0)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 image0)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 star2)
	(supports instrument16 image0)
	(supports instrument16 thermograph1)
	(calibration_target instrument16 star2)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet3)
	(supports instrument17 thermograph1)
	(supports instrument17 image0)
	(calibration_target instrument17 star2)
	(on_board instrument17 satellite5)
	(power_avail satellite5)
	(pointing satellite5 groundstation1)
)
(:goal (and
	(have_image planet3 image0)
	(have_image phenomenon4 image0)
	(have_image planet5 image0)
))

)
