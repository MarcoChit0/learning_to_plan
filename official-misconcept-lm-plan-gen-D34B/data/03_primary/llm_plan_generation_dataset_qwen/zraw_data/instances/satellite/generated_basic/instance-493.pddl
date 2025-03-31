(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite1 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite2 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite3 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	image0 - mode
	spectrograph3 - mode
	thermograph1 - mode
	spectrograph2 - mode
	star1 - direction
	star4 - direction
	groundstation2 - direction
	groundstation3 - direction
	groundstation0 - direction
	phenomenon5 - direction
	star6 - direction
	star7 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star1)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 star1)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument6 image0)
	(supports instrument6 thermograph1)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 thermograph1)
	(supports instrument7 image0)
	(calibration_target instrument7 star1)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 star4)
	(supports instrument9 spectrograph3)
	(calibration_target instrument9 star4)
	(supports instrument10 image0)
	(calibration_target instrument10 star4)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 groundstation2)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(on_board instrument11 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument12 image0)
	(supports instrument12 spectrograph2)
	(calibration_target instrument12 star4)
	(supports instrument13 image0)
	(calibration_target instrument13 groundstation0)
	(supports instrument14 spectrograph3)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star6)
	(supports instrument15 image0)
	(supports instrument15 spectrograph2)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 groundstation2)
	(supports instrument16 thermograph1)
	(supports instrument16 image0)
	(supports instrument16 spectrograph2)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 spectrograph2)
	(supports instrument17 thermograph1)
	(supports instrument17 spectrograph3)
	(calibration_target instrument17 groundstation3)
	(supports instrument18 thermograph1)
	(supports instrument18 spectrograph2)
	(supports instrument18 image0)
	(calibration_target instrument18 groundstation0)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(on_board instrument18 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon5)
)
(:goal (and
	(pointing satellite1 groundstation3)
	(have_image phenomenon5 spectrograph2)
	(have_image star6 spectrograph3)
	(have_image star7 thermograph1)
))

)
