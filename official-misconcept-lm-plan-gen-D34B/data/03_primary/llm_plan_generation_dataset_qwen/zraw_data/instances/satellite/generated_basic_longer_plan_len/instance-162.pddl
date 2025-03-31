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
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite4 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite5 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	satellite6 - satellite
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	infrared1 - mode
	thermograph0 - mode
	star0 - direction
	star1 - direction
	planet2 - direction
	planet3 - direction
	star4 - direction
	phenomenon5 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 star0)
	(supports instrument1 thermograph0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument4 thermograph0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star0)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument5 thermograph0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star0)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet2)
	(supports instrument6 thermograph0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star0)
	(supports instrument7 infrared1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star0)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet3)
	(supports instrument8 thermograph0)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared1)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star0)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet2)
	(supports instrument10 infrared1)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 star0)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star0)
	(supports instrument12 thermograph0)
	(supports instrument12 infrared1)
	(calibration_target instrument12 star0)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 star0)
	(supports instrument14 infrared1)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 star0)
	(supports instrument15 infrared1)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 star0)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 star0)
	(on_board instrument10 satellite5)
	(on_board instrument11 satellite5)
	(on_board instrument12 satellite5)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(on_board instrument15 satellite5)
	(on_board instrument16 satellite5)
	(power_avail satellite5)
	(pointing satellite5 planet6)
	(supports instrument17 infrared1)
	(calibration_target instrument17 star0)
	(supports instrument18 infrared1)
	(supports instrument18 thermograph0)
	(calibration_target instrument18 star0)
	(supports instrument19 infrared1)
	(supports instrument19 thermograph0)
	(calibration_target instrument19 star0)
	(supports instrument20 thermograph0)
	(supports instrument20 infrared1)
	(calibration_target instrument20 star0)
	(on_board instrument17 satellite6)
	(on_board instrument18 satellite6)
	(on_board instrument19 satellite6)
	(on_board instrument20 satellite6)
	(power_avail satellite6)
	(pointing satellite6 star0)
)
(:goal (and
	(pointing satellite1 planet2)
	(pointing satellite2 phenomenon5)
	(pointing satellite3 star7)
	(pointing satellite6 phenomenon5)
	(have_image star1 thermograph0)
	(have_image planet2 thermograph0)
	(have_image planet3 thermograph0)
	(have_image star4 infrared1)
	(have_image phenomenon5 thermograph0)
	(have_image planet6 thermograph0)
	(have_image star7 infrared1)
	(have_image planet8 thermograph0)
))

)
