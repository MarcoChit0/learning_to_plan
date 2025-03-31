(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	thermograph1 - mode
	image0 - mode
	star0 - direction
	groundstation3 - direction
	groundstation2 - direction
	star1 - direction
	groundstation4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 image0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star0)
	(supports instrument2 image0)
	(calibration_target instrument2 star1)
	(supports instrument3 thermograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 image0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 thermograph1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument6 image0)
	(calibration_target instrument6 star0)
	(supports instrument7 image0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation4)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument8 image0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star0)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 thermograph1)
	(supports instrument12 image0)
	(calibration_target instrument12 star1)
	(supports instrument13 image0)
	(supports instrument13 thermograph1)
	(calibration_target instrument13 groundstation4)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
)
(:goal (and
	(pointing satellite2 planet5)
	(have_image planet5 image0)
))

)
