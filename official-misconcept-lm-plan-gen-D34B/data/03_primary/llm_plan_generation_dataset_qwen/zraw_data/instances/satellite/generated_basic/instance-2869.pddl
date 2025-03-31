(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
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
	thermograph1 - mode
	image0 - mode
	groundstation4 - direction
	star2 - direction
	star0 - direction
	groundstation1 - direction
	groundstation3 - direction
	star5 - direction
	star6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 groundstation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon7)
	(supports instrument4 image0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(supports instrument5 image0)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 image0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 image0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star2)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation4)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument9 thermograph1)
	(supports instrument9 image0)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 thermograph1)
	(supports instrument10 image0)
	(calibration_target instrument10 star2)
	(supports instrument11 image0)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 star0)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 image0)
	(supports instrument13 thermograph1)
	(calibration_target instrument13 groundstation3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon7)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite1 groundstation3)
	(have_image star5 image0)
	(have_image star6 image0)
	(have_image phenomenon7 thermograph1)
))

)
