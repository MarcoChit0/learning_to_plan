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
	instrument9 - instrument
	instrument10 - instrument
	satellite2 - satellite
	instrument11 - instrument
	instrument12 - instrument
	satellite3 - satellite
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	instrument19 - instrument
	thermograph1 - mode
	image3 - mode
	spectrograph2 - mode
	image0 - mode
	star3 - direction
	star2 - direction
	star0 - direction
	groundstation1 - direction
	planet4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 star2)
	(supports instrument1 image3)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 thermograph1)
	(supports instrument2 image0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star3)
	(supports instrument3 thermograph1)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image0)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument5 spectrograph2)
	(supports instrument5 thermograph1)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 image0)
	(supports instrument6 thermograph1)
	(supports instrument6 image3)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 image0)
	(supports instrument7 thermograph1)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 star3)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 star2)
	(supports instrument10 image3)
	(calibration_target instrument10 star3)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(on_board instrument10 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument11 spectrograph2)
	(supports instrument11 thermograph1)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 spectrograph2)
	(supports instrument12 thermograph1)
	(supports instrument12 image3)
	(calibration_target instrument12 star3)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument13 spectrograph2)
	(calibration_target instrument13 star2)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet5)
	(supports instrument14 image0)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 star0)
	(supports instrument16 image0)
	(supports instrument16 thermograph1)
	(supports instrument16 spectrograph2)
	(calibration_target instrument16 star0)
	(supports instrument17 image0)
	(calibration_target instrument17 star0)
	(supports instrument18 image3)
	(supports instrument18 spectrograph2)
	(calibration_target instrument18 star0)
	(supports instrument19 spectrograph2)
	(supports instrument19 image0)
	(supports instrument19 image3)
	(calibration_target instrument19 groundstation1)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(on_board instrument19 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet4)
)
(:goal (and
	(pointing satellite0 star3)
	(pointing satellite1 star2)
	(pointing satellite3 star0)
	(have_image planet4 thermograph1)
	(have_image planet5 image0)
))

)
