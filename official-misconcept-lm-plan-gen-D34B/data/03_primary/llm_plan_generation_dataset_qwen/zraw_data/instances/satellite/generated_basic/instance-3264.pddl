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
	instrument15 - instrument
	satellite3 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	image0 - mode
	thermograph2 - mode
	spectrograph1 - mode
	star2 - direction
	star1 - direction
	groundstation0 - direction
	star3 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 star2)
	(supports instrument2 thermograph2)
	(supports instrument2 image0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image0)
	(supports instrument4 thermograph2)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument6 image0)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 thermograph2)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 star1)
	(supports instrument8 spectrograph1)
	(supports instrument8 image0)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 star2)
	(supports instrument9 thermograph2)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 star2)
	(supports instrument10 spectrograph1)
	(supports instrument10 image0)
	(supports instrument10 thermograph2)
	(calibration_target instrument10 star1)
	(supports instrument11 image0)
	(supports instrument11 thermograph2)
	(supports instrument11 spectrograph1)
	(calibration_target instrument11 star1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(on_board instrument11 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument12 image0)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 spectrograph1)
	(supports instrument13 image0)
	(calibration_target instrument13 star2)
	(supports instrument14 thermograph2)
	(supports instrument14 image0)
	(supports instrument14 spectrograph1)
	(calibration_target instrument14 star2)
	(supports instrument15 image0)
	(supports instrument15 spectrograph1)
	(calibration_target instrument15 groundstation0)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(on_board instrument15 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument16 spectrograph1)
	(supports instrument16 thermograph2)
	(calibration_target instrument16 star2)
	(supports instrument17 thermograph2)
	(supports instrument17 spectrograph1)
	(supports instrument17 image0)
	(calibration_target instrument17 star2)
	(supports instrument18 thermograph2)
	(supports instrument18 image0)
	(supports instrument18 spectrograph1)
	(calibration_target instrument18 star1)
	(supports instrument19 image0)
	(supports instrument19 thermograph2)
	(supports instrument19 spectrograph1)
	(calibration_target instrument19 groundstation0)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(on_board instrument18 satellite3)
	(on_board instrument19 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
)
(:goal (and
	(pointing satellite2 star3)
	(have_image star3 spectrograph1)
))

)
