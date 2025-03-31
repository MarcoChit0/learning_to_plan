(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite1 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	image3 - mode
	spectrograph0 - mode
	spectrograph1 - mode
	thermograph2 - mode
	groundstation0 - direction
	star2 - direction
	groundstation3 - direction
	groundstation1 - direction
	phenomenon4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 spectrograph1)
	(supports instrument1 spectrograph0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 image3)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 image3)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image3)
	(calibration_target instrument4 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 image3)
	(supports instrument6 spectrograph1)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 thermograph2)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 thermograph2)
	(supports instrument8 image3)
	(calibration_target instrument8 star2)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon4)
	(supports instrument9 image3)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 image3)
	(calibration_target instrument10 groundstation0)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument11 spectrograph0)
	(supports instrument11 thermograph2)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 image3)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 spectrograph0)
	(supports instrument13 thermograph2)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 spectrograph0)
	(supports instrument14 thermograph2)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 thermograph2)
	(calibration_target instrument15 star2)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 groundstation1)
	(supports instrument17 image3)
	(supports instrument17 thermograph2)
	(calibration_target instrument17 groundstation3)
	(supports instrument18 spectrograph1)
	(supports instrument18 image3)
	(supports instrument18 thermograph2)
	(calibration_target instrument18 groundstation1)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation3)
)
(:goal (and
	(pointing satellite4 phenomenon5)
	(have_image phenomenon4 spectrograph0)
	(have_image phenomenon5 thermograph2)
))

)
