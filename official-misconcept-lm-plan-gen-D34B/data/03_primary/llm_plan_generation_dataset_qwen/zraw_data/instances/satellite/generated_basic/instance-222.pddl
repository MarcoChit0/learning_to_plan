(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
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
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	thermograph0 - mode
	image1 - mode
	groundstation3 - direction
	star0 - direction
	star4 - direction
	groundstation2 - direction
	star1 - direction
	star5 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star1)
	(supports instrument1 image1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 star0)
	(supports instrument2 image1)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star7)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 image1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star4)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet8)
	(supports instrument5 thermograph0)
	(supports instrument5 image1)
	(calibration_target instrument5 star1)
	(supports instrument6 image1)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 image1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star1)
	(supports instrument8 image1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star0)
	(supports instrument9 image1)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 groundstation3)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument10 image1)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation2)
	(supports instrument11 image1)
	(calibration_target instrument11 star1)
	(supports instrument12 image1)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star0)
	(supports instrument13 thermograph0)
	(supports instrument13 image1)
	(calibration_target instrument13 star4)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument14 image1)
	(calibration_target instrument14 groundstation2)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 groundstation2)
	(supports instrument16 image1)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 star1)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation3)
)
(:goal (and
	(pointing satellite1 groundstation2)
	(pointing satellite2 planet8)
	(pointing satellite3 star6)
	(have_image star5 thermograph0)
	(have_image star6 thermograph0)
	(have_image star7 image1)
	(have_image planet8 thermograph0)
))

)
