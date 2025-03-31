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
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	spectrograph2 - mode
	thermograph3 - mode
	thermograph0 - mode
	thermograph1 - mode
	groundstation0 - direction
	groundstation1 - direction
	groundstation2 - direction
	star3 - direction
	phenomenon4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 thermograph1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 star3)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 thermograph1)
	(supports instrument4 thermograph3)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 spectrograph2)
	(supports instrument5 thermograph0)
	(supports instrument5 thermograph3)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 thermograph3)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 groundstation2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument7 thermograph3)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 spectrograph2)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 thermograph0)
	(supports instrument9 thermograph3)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 groundstation0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument12 thermograph1)
	(supports instrument12 thermograph0)
	(supports instrument12 spectrograph2)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 spectrograph2)
	(supports instrument13 thermograph3)
	(calibration_target instrument13 star3)
	(supports instrument14 thermograph0)
	(supports instrument14 thermograph3)
	(calibration_target instrument14 star3)
	(supports instrument15 spectrograph2)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 groundstation1)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet5)
	(supports instrument16 thermograph0)
	(supports instrument16 spectrograph2)
	(calibration_target instrument16 star3)
	(supports instrument17 thermograph0)
	(supports instrument17 thermograph3)
	(calibration_target instrument17 groundstation2)
	(supports instrument18 spectrograph2)
	(supports instrument18 thermograph0)
	(supports instrument18 thermograph1)
	(calibration_target instrument18 star3)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon4)
)
(:goal (and
	(have_image phenomenon4 thermograph1)
	(have_image planet5 thermograph1)
))

)
