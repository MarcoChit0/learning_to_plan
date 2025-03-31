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
	satellite2 - satellite
	instrument8 - instrument
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
	thermograph2 - mode
	spectrograph0 - mode
	spectrograph3 - mode
	image1 - mode
	groundstation1 - direction
	star0 - direction
	planet2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph0)
	(supports instrument2 image1)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument4 spectrograph0)
	(supports instrument4 thermograph2)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 spectrograph0)
	(supports instrument5 image1)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 star0)
	(supports instrument6 spectrograph0)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 star0)
	(supports instrument7 spectrograph3)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 groundstation1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument8 thermograph2)
	(supports instrument8 spectrograph0)
	(supports instrument8 image1)
	(calibration_target instrument8 star0)
	(supports instrument9 spectrograph0)
	(supports instrument9 image1)
	(calibration_target instrument9 star0)
	(supports instrument10 spectrograph3)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 image1)
	(supports instrument11 spectrograph3)
	(calibration_target instrument11 star0)
	(supports instrument12 thermograph2)
	(supports instrument12 spectrograph3)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 groundstation1)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon3)
	(supports instrument13 image1)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 spectrograph3)
	(calibration_target instrument14 star0)
	(supports instrument15 image1)
	(supports instrument15 spectrograph0)
	(supports instrument15 spectrograph3)
	(calibration_target instrument15 groundstation1)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 star0)
	(supports instrument17 spectrograph3)
	(supports instrument17 image1)
	(calibration_target instrument17 star0)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument18 spectrograph3)
	(supports instrument18 spectrograph0)
	(supports instrument18 image1)
	(calibration_target instrument18 groundstation1)
	(supports instrument19 spectrograph3)
	(supports instrument19 image1)
	(calibration_target instrument19 star0)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
)
(:goal (and
	(pointing satellite1 star0)
	(pointing satellite2 planet2)
	(pointing satellite3 star0)
	(have_image planet2 spectrograph3)
	(have_image phenomenon3 image1)
))

)
