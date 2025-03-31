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
	satellite5 - satellite
	instrument18 - instrument
	instrument19 - instrument
	instrument20 - instrument
	instrument21 - instrument
	infrared1 - mode
	spectrograph2 - mode
	thermograph3 - mode
	thermograph0 - mode
	star1 - direction
	star0 - direction
	star2 - direction
	phenomenon3 - direction
	planet4 - direction
	star5 - direction
	phenomenon6 - direction
	star7 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 spectrograph2)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon3)
	(supports instrument1 thermograph3)
	(supports instrument1 infrared1)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 star2)
	(supports instrument3 spectrograph2)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared1)
	(supports instrument4 thermograph3)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared1)
	(supports instrument5 thermograph3)
	(calibration_target instrument5 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument6 infrared1)
	(supports instrument6 thermograph3)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 star1)
	(supports instrument7 spectrograph2)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star2)
	(supports instrument8 infrared1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star1)
	(supports instrument9 thermograph0)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 star0)
	(supports instrument10 infrared1)
	(calibration_target instrument10 star2)
	(supports instrument11 infrared1)
	(supports instrument11 thermograph3)
	(calibration_target instrument11 star0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
	(supports instrument12 thermograph3)
	(supports instrument12 infrared1)
	(supports instrument12 spectrograph2)
	(calibration_target instrument12 star2)
	(supports instrument13 thermograph3)
	(supports instrument13 spectrograph2)
	(supports instrument13 thermograph0)
	(calibration_target instrument13 star1)
	(supports instrument14 thermograph3)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 star2)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star7)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 star1)
	(supports instrument16 spectrograph2)
	(calibration_target instrument16 star2)
	(supports instrument17 thermograph3)
	(supports instrument17 spectrograph2)
	(calibration_target instrument17 star1)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon6)
	(supports instrument18 thermograph3)
	(supports instrument18 infrared1)
	(supports instrument18 spectrograph2)
	(calibration_target instrument18 star1)
	(supports instrument19 thermograph3)
	(supports instrument19 infrared1)
	(supports instrument19 thermograph0)
	(calibration_target instrument19 star0)
	(supports instrument20 infrared1)
	(supports instrument20 thermograph0)
	(calibration_target instrument20 star2)
	(supports instrument21 thermograph3)
	(calibration_target instrument21 star2)
	(on_board instrument18 satellite5)
	(on_board instrument19 satellite5)
	(on_board instrument20 satellite5)
	(on_board instrument21 satellite5)
	(power_avail satellite5)
	(pointing satellite5 phenomenon3)
)
(:goal (and
	(pointing satellite1 phenomenon3)
	(have_image phenomenon3 spectrograph2)
	(have_image planet4 spectrograph2)
	(have_image star5 thermograph3)
	(have_image phenomenon6 spectrograph2)
	(have_image star7 spectrograph2)
))

)
