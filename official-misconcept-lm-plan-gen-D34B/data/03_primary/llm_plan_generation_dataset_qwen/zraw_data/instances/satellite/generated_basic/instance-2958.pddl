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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	satellite4 - satellite
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	spectrograph2 - mode
	image1 - mode
	thermograph0 - mode
	thermograph3 - mode
	groundstation1 - direction
	groundstation0 - direction
	star2 - direction
	groundstation3 - direction
	phenomenon4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 image1)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 star2)
	(supports instrument2 thermograph0)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 thermograph0)
	(supports instrument3 spectrograph2)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet5)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 image1)
	(supports instrument5 spectrograph2)
	(supports instrument5 thermograph3)
	(calibration_target instrument5 star2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument6 image1)
	(supports instrument6 thermograph3)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 star2)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 groundstation3)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 spectrograph2)
	(supports instrument10 image1)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 spectrograph2)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star2)
	(supports instrument12 spectrograph2)
	(calibration_target instrument12 groundstation0)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
	(supports instrument13 thermograph3)
	(calibration_target instrument13 star2)
	(supports instrument14 thermograph3)
	(calibration_target instrument14 groundstation0)
	(supports instrument15 thermograph3)
	(calibration_target instrument15 star2)
	(supports instrument16 image1)
	(supports instrument16 thermograph0)
	(supports instrument16 spectrograph2)
	(calibration_target instrument16 groundstation3)
	(supports instrument17 thermograph3)
	(supports instrument17 spectrograph2)
	(calibration_target instrument17 star2)
	(supports instrument18 thermograph0)
	(supports instrument18 spectrograph2)
	(supports instrument18 thermograph3)
	(calibration_target instrument18 groundstation3)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet5)
)
(:goal (and
	(pointing satellite2 planet5)
	(have_image phenomenon4 image1)
	(have_image planet5 spectrograph2)
))

)
