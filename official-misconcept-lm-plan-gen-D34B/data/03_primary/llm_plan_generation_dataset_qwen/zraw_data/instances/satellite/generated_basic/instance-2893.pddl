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
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite2 - satellite
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	satellite4 - satellite
	instrument18 - instrument
	instrument19 - instrument
	thermograph0 - mode
	star3 - direction
	groundstation1 - direction
	star4 - direction
	star2 - direction
	groundstation5 - direction
	star0 - direction
	planet6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star3)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation5)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star2)
	(calibration_target instrument2 star3)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation1)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon8)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 star4)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation1)
	(calibration_target instrument5 star2)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation1)
	(calibration_target instrument7 star3)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation1)
	(calibration_target instrument8 star4)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon8)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 star0)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star2)
	(calibration_target instrument11 groundstation5)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star4)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 groundstation5)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 groundstation1)
	(calibration_target instrument15 groundstation5)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 star4)
	(supports instrument17 thermograph0)
	(calibration_target instrument17 star2)
	(calibration_target instrument17 groundstation5)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument18 thermograph0)
	(calibration_target instrument18 groundstation5)
	(supports instrument19 thermograph0)
	(calibration_target instrument19 star0)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet6)
)
(:goal (and
	(pointing satellite1 phenomenon8)
	(pointing satellite4 star0)
	(have_image planet6 thermograph0)
	(have_image phenomenon7 thermograph0)
	(have_image phenomenon8 thermograph0)
))

)
