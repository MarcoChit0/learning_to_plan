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
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	image2 - mode
	thermograph0 - mode
	image1 - mode
	groundstation1 - direction
	star0 - direction
	planet2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image1)
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet2)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(supports instrument2 image2)
	(supports instrument2 thermograph0)
	(supports instrument2 image1)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 image2)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image2)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument7 image2)
	(supports instrument7 image1)
	(calibration_target instrument7 star0)
	(supports instrument8 image2)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 image1)
	(supports instrument9 image2)
	(calibration_target instrument9 star0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument10 image1)
	(supports instrument10 image2)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 thermograph0)
	(supports instrument11 image1)
	(calibration_target instrument11 star0)
	(supports instrument12 image2)
	(supports instrument12 thermograph0)
	(supports instrument12 image1)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 image2)
	(supports instrument13 image1)
	(calibration_target instrument13 star0)
	(supports instrument14 image2)
	(supports instrument14 thermograph0)
	(supports instrument14 image1)
	(calibration_target instrument14 star0)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite1 planet2)
	(pointing satellite2 groundstation1)
	(have_image planet2 image1)
	(have_image planet3 image1)
))

)
