(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	satellite4 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	thermograph0 - mode
	image1 - mode
	star0 - direction
	planet1 - direction
	phenomenon2 - direction
	star3 - direction
	phenomenon4 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument2 image1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star0)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument4 thermograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 star0)
	(supports instrument5 image1)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon2)
	(supports instrument6 image1)
	(calibration_target instrument6 star0)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
	(supports instrument7 thermograph0)
	(supports instrument7 image1)
	(calibration_target instrument7 star0)
	(supports instrument8 thermograph0)
	(supports instrument8 image1)
	(calibration_target instrument8 star0)
	(supports instrument9 thermograph0)
	(supports instrument9 image1)
	(calibration_target instrument9 star0)
	(on_board instrument7 satellite4)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon4)
)
(:goal (and
	(pointing satellite2 star3)
	(pointing satellite4 phenomenon2)
	(have_image planet1 image1)
	(have_image phenomenon2 image1)
	(have_image star3 thermograph0)
	(have_image phenomenon4 image1)
))

)
