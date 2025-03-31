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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	satellite4 - satellite
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	image0 - mode
	spectrograph1 - mode
	thermograph2 - mode
	groundstation1 - direction
	groundstation0 - direction
	star3 - direction
	star2 - direction
	planet4 - direction
	phenomenon5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 thermograph2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 image0)
	(supports instrument1 spectrograph1)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star2)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 thermograph2)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star3)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument5 spectrograph1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument6 spectrograph1)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 star2)
	(supports instrument8 spectrograph1)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 image0)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 groundstation1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 star2)
	(supports instrument11 spectrograph1)
	(supports instrument11 thermograph2)
	(supports instrument11 image0)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 image0)
	(supports instrument12 thermograph2)
	(supports instrument12 spectrograph1)
	(calibration_target instrument12 star2)
	(supports instrument13 thermograph2)
	(calibration_target instrument13 star2)
	(supports instrument14 thermograph2)
	(supports instrument14 spectrograph1)
	(calibration_target instrument14 groundstation0)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
	(supports instrument15 thermograph2)
	(supports instrument15 image0)
	(calibration_target instrument15 star2)
	(supports instrument16 thermograph2)
	(supports instrument16 spectrograph1)
	(supports instrument16 image0)
	(calibration_target instrument16 star3)
	(supports instrument17 spectrograph1)
	(supports instrument17 image0)
	(supports instrument17 thermograph2)
	(calibration_target instrument17 star3)
	(supports instrument18 image0)
	(calibration_target instrument18 star2)
	(supports instrument19 image0)
	(supports instrument19 thermograph2)
	(supports instrument19 spectrograph1)
	(calibration_target instrument19 star2)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star3)
)
(:goal (and
	(pointing satellite2 star2)
	(have_image planet4 spectrograph1)
	(have_image phenomenon5 image0)
	(have_image planet6 thermograph2)
))

)
