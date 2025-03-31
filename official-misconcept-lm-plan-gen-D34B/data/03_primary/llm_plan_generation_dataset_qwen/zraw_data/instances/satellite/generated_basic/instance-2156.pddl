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
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	satellite5 - satellite
	instrument17 - instrument
	instrument18 - instrument
	thermograph2 - mode
	image3 - mode
	spectrograph0 - mode
	thermograph1 - mode
	star0 - direction
	star2 - direction
	groundstation1 - direction
	star3 - direction
	planet4 - direction
	phenomenon5 - direction
	planet6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star2)
	(supports instrument1 image3)
	(supports instrument1 thermograph2)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 star0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument4 thermograph2)
	(supports instrument4 image3)
	(supports instrument4 thermograph1)
	(calibration_target instrument4 star0)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 star2)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument6 image3)
	(supports instrument6 thermograph1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star0)
	(supports instrument7 spectrograph0)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 star0)
	(supports instrument8 thermograph2)
	(supports instrument8 image3)
	(calibration_target instrument8 star2)
	(supports instrument9 spectrograph0)
	(supports instrument9 image3)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 star0)
	(supports instrument10 spectrograph0)
	(supports instrument10 thermograph1)
	(supports instrument10 thermograph2)
	(calibration_target instrument10 groundstation1)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet7)
	(supports instrument11 thermograph1)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 star2)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
	(supports instrument12 thermograph2)
	(supports instrument12 image3)
	(calibration_target instrument12 star0)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 thermograph2)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 thermograph2)
	(supports instrument15 thermograph1)
	(calibration_target instrument15 groundstation1)
	(supports instrument16 thermograph1)
	(supports instrument16 thermograph2)
	(calibration_target instrument16 star2)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(on_board instrument16 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet4)
	(supports instrument17 spectrograph0)
	(supports instrument17 image3)
	(supports instrument17 thermograph2)
	(calibration_target instrument17 groundstation1)
	(supports instrument18 thermograph1)
	(calibration_target instrument18 groundstation1)
	(on_board instrument17 satellite5)
	(on_board instrument18 satellite5)
	(power_avail satellite5)
	(pointing satellite5 planet4)
)
(:goal (and
	(pointing satellite1 planet8)
	(pointing satellite4 planet4)
	(have_image star3 thermograph1)
	(have_image planet4 thermograph1)
	(have_image phenomenon5 image3)
	(have_image planet6 spectrograph0)
	(have_image planet7 image3)
	(have_image planet8 thermograph1)
))

)
