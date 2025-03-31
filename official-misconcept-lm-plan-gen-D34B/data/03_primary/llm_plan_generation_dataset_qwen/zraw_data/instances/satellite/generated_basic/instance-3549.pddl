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
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	thermograph1 - mode
	image0 - mode
	star1 - direction
	star2 - direction
	groundstation3 - direction
	star0 - direction
	planet4 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph1)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph1)
	(supports instrument2 image0)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star1)
	(supports instrument5 thermograph1)
	(supports instrument5 image0)
	(calibration_target instrument5 star1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument6 thermograph1)
	(supports instrument6 image0)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star0)
	(supports instrument8 image0)
	(calibration_target instrument8 star2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet4)
	(supports instrument9 image0)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 image0)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 star0)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
)
(:goal (and
	(pointing satellite1 star2)
	(pointing satellite2 star2)
	(pointing satellite3 planet4)
	(have_image planet4 image0)
))

)
