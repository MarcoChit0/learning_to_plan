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
	satellite3 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	satellite4 - satellite
	instrument18 - instrument
	instrument19 - instrument
	thermograph1 - mode
	image0 - mode
	spectrograph2 - mode
	star0 - direction
	groundstation3 - direction
	star2 - direction
	groundstation1 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph1)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 spectrograph2)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 spectrograph2)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument4 image0)
	(calibration_target instrument4 star0)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 spectrograph2)
	(supports instrument6 image0)
	(supports instrument6 thermograph1)
	(calibration_target instrument6 star0)
	(supports instrument7 thermograph1)
	(supports instrument7 spectrograph2)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 star2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument9 image0)
	(calibration_target instrument9 star2)
	(supports instrument10 image0)
	(supports instrument10 spectrograph2)
	(supports instrument10 thermograph1)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation3)
	(supports instrument12 spectrograph2)
	(supports instrument12 image0)
	(calibration_target instrument12 star0)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
	(supports instrument13 thermograph1)
	(supports instrument13 spectrograph2)
	(supports instrument13 image0)
	(calibration_target instrument13 star0)
	(supports instrument14 thermograph1)
	(calibration_target instrument14 groundstation3)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 star2)
	(supports instrument16 thermograph1)
	(supports instrument16 image0)
	(supports instrument16 spectrograph2)
	(calibration_target instrument16 groundstation3)
	(supports instrument17 spectrograph2)
	(supports instrument17 image0)
	(supports instrument17 thermograph1)
	(calibration_target instrument17 groundstation1)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
	(supports instrument18 image0)
	(supports instrument18 thermograph1)
	(supports instrument18 spectrograph2)
	(calibration_target instrument18 star2)
	(supports instrument19 spectrograph2)
	(supports instrument19 thermograph1)
	(supports instrument19 image0)
	(calibration_target instrument19 groundstation1)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star2)
)
(:goal (and
	(pointing satellite2 star0)
	(pointing satellite3 groundstation1)
	(have_image phenomenon4 image0)
	(have_image phenomenon5 image0)
	(have_image phenomenon6 thermograph1)
))

)
