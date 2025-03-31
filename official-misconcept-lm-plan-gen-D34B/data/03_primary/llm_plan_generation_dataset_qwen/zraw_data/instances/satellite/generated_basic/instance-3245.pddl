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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	image1 - mode
	thermograph0 - mode
	star0 - direction
	star1 - direction
	groundstation2 - direction
	phenomenon3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star0)
	(supports instrument1 image1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 thermograph0)
	(supports instrument2 image1)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 image1)
	(calibration_target instrument6 star0)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star0)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon4)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star0)
	(supports instrument9 image1)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star1)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument10 thermograph0)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation2)
	(on_board instrument10 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon4)
)
(:goal (and
	(pointing satellite1 phenomenon3)
	(pointing satellite2 phenomenon4)
	(pointing satellite3 star1)
	(pointing satellite4 star1)
	(have_image phenomenon3 thermograph0)
	(have_image phenomenon4 thermograph0)
))

)
