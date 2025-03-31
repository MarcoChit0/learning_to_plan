(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
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
	image0 - mode
	thermograph1 - mode
	groundstation3 - direction
	groundstation4 - direction
	star2 - direction
	star0 - direction
	groundstation1 - direction
	planet5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 thermograph1)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 thermograph1)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet5)
	(supports instrument4 image0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star2)
	(supports instrument5 thermograph1)
	(supports instrument5 image0)
	(calibration_target instrument5 star2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument6 thermograph1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation1)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument7 image0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation4)
	(supports instrument8 image0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star2)
	(supports instrument9 thermograph1)
	(supports instrument9 image0)
	(calibration_target instrument9 star2)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 star0)
	(supports instrument11 image0)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 groundstation1)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite0 phenomenon7)
	(pointing satellite1 groundstation3)
	(have_image planet5 thermograph1)
	(have_image phenomenon6 thermograph1)
	(have_image phenomenon7 thermograph1)
))

)
