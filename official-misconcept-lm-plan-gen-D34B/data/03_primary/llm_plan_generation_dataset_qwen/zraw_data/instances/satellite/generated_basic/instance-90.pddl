(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
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
	satellite4 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	thermograph0 - mode
	thermograph1 - mode
	star4 - direction
	groundstation5 - direction
	groundstation1 - direction
	star0 - direction
	star2 - direction
	star3 - direction
	phenomenon6 - direction
	planet7 - direction
	phenomenon8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 thermograph0)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 groundstation1)
	(calibration_target instrument4 star2)
	(supports instrument5 thermograph0)
	(calibration_target instrument5 groundstation1)
	(calibration_target instrument5 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon8)
	(supports instrument6 thermograph1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star3)
	(calibration_target instrument7 star4)
	(supports instrument8 thermograph0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star2)
	(supports instrument9 thermograph1)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star3)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 star2)
	(calibration_target instrument10 star0)
	(supports instrument11 thermograph1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon9)
	(supports instrument12 thermograph1)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 star4)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 groundstation5)
	(calibration_target instrument13 star0)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument14 thermograph1)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 groundstation1)
	(calibration_target instrument14 groundstation5)
	(supports instrument15 thermograph0)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 star3)
	(supports instrument16 thermograph1)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 star3)
	(calibration_target instrument16 star0)
	(supports instrument17 thermograph0)
	(supports instrument17 thermograph1)
	(calibration_target instrument17 star2)
	(supports instrument18 thermograph0)
	(calibration_target instrument18 star3)
	(supports instrument19 thermograph1)
	(supports instrument19 thermograph0)
	(calibration_target instrument19 star3)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon8)
)
(:goal (and
	(pointing satellite0 phenomenon8)
	(pointing satellite1 phenomenon8)
	(pointing satellite4 planet7)
	(have_image phenomenon6 thermograph1)
	(have_image planet7 thermograph1)
	(have_image phenomenon8 thermograph0)
	(have_image phenomenon9 thermograph1)
))

)
