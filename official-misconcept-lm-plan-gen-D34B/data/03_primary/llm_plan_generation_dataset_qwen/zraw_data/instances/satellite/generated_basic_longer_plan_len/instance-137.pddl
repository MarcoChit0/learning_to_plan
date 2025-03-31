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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	thermograph0 - mode
	groundstation3 - direction
	star1 - direction
	star0 - direction
	star2 - direction
	planet4 - direction
	planet5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
	star8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 star2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star0)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star0)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation3)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation3)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon7)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star0)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 star1)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 star0)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 star0)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 star2)
	(supports instrument17 thermograph0)
	(calibration_target instrument17 star2)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon6)
)
(:goal (and
	(pointing satellite2 groundstation3)
	(pointing satellite3 planet4)
	(pointing satellite4 planet4)
	(have_image planet4 thermograph0)
	(have_image planet5 thermograph0)
	(have_image phenomenon6 thermograph0)
	(have_image phenomenon7 thermograph0)
	(have_image star8 thermograph0)
))

)
