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
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite3 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	satellite4 - satellite
	instrument20 - instrument
	thermograph0 - mode
	spectrograph1 - mode
	star1 - direction
	groundstation3 - direction
	groundstation0 - direction
	star2 - direction
	planet4 - direction
	star5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 star1)
	(supports instrument1 thermograph0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 star2)
	(supports instrument2 thermograph0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star1)
	(supports instrument3 spectrograph1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star2)
	(supports instrument5 thermograph0)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star2)
	(supports instrument7 spectrograph1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star2)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet4)
	(supports instrument9 spectrograph1)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star1)
	(supports instrument10 spectrograph1)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 star1)
	(supports instrument11 thermograph0)
	(supports instrument11 spectrograph1)
	(calibration_target instrument11 star1)
	(supports instrument12 spectrograph1)
	(supports instrument12 thermograph0)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 thermograph0)
	(supports instrument13 spectrograph1)
	(calibration_target instrument13 star2)
	(supports instrument14 spectrograph1)
	(supports instrument14 thermograph0)
	(calibration_target instrument14 groundstation0)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
	(supports instrument15 spectrograph1)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 spectrograph1)
	(supports instrument16 thermograph0)
	(calibration_target instrument16 star2)
	(supports instrument17 thermograph0)
	(supports instrument17 spectrograph1)
	(calibration_target instrument17 groundstation3)
	(supports instrument18 spectrograph1)
	(calibration_target instrument18 groundstation3)
	(supports instrument19 spectrograph1)
	(calibration_target instrument19 groundstation0)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(on_board instrument18 satellite3)
	(on_board instrument19 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
	(supports instrument20 spectrograph1)
	(supports instrument20 thermograph0)
	(calibration_target instrument20 star2)
	(on_board instrument20 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star5)
)
(:goal (and
	(pointing satellite2 star5)
	(pointing satellite3 planet4)
	(have_image planet4 spectrograph1)
	(have_image star5 spectrograph1)
	(have_image phenomenon6 spectrograph1)
))

)
