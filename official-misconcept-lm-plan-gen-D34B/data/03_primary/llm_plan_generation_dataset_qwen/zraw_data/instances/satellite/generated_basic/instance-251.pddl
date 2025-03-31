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
	satellite2 - satellite
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	thermograph1 - mode
	image0 - mode
	groundstation3 - direction
	star4 - direction
	groundstation1 - direction
	groundstation2 - direction
	star0 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 thermograph1)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(supports instrument2 image0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star4)
	(supports instrument3 thermograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument5 image0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star0)
	(supports instrument8 image0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation2)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet8)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 star4)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star7)
	(supports instrument10 thermograph1)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 thermograph1)
	(supports instrument11 image0)
	(calibration_target instrument11 star0)
	(supports instrument12 thermograph1)
	(supports instrument12 image0)
	(calibration_target instrument12 groundstation2)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
	(supports instrument13 image0)
	(supports instrument13 thermograph1)
	(calibration_target instrument13 star0)
	(on_board instrument13 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon6)
)
(:goal (and
	(have_image phenomenon5 image0)
	(have_image phenomenon6 thermograph1)
	(have_image star7 thermograph1)
	(have_image planet8 thermograph1)
))

)
