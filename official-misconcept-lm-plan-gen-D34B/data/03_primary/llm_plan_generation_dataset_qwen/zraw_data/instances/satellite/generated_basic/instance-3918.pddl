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
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite4 - satellite
	instrument16 - instrument
	instrument17 - instrument
	instrument18 - instrument
	spectrograph0 - mode
	spectrograph2 - mode
	spectrograph1 - mode
	thermograph3 - mode
	star5 - direction
	star2 - direction
	star0 - direction
	groundstation4 - direction
	groundstation1 - direction
	star3 - direction
	star6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 spectrograph0)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument1 spectrograph0)
	(supports instrument1 spectrograph1)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 star3)
	(supports instrument2 spectrograph1)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star3)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph2)
	(supports instrument3 thermograph3)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star0)
	(supports instrument4 thermograph3)
	(supports instrument4 spectrograph0)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 star0)
	(supports instrument5 thermograph3)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 spectrograph0)
	(supports instrument6 thermograph3)
	(calibration_target instrument6 star5)
	(calibration_target instrument6 groundstation1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet8)
	(supports instrument7 thermograph3)
	(supports instrument7 spectrograph0)
	(supports instrument7 spectrograph2)
	(calibration_target instrument7 star3)
	(supports instrument8 spectrograph2)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation4)
	(supports instrument9 thermograph3)
	(calibration_target instrument9 star3)
	(calibration_target instrument9 groundstation4)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
	(supports instrument10 spectrograph0)
	(supports instrument10 thermograph3)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation1)
	(calibration_target instrument11 star5)
	(supports instrument12 spectrograph2)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 groundstation4)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 spectrograph1)
	(supports instrument13 thermograph3)
	(calibration_target instrument13 star2)
	(supports instrument14 spectrograph1)
	(supports instrument14 spectrograph2)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 star0)
	(supports instrument15 spectrograph2)
	(calibration_target instrument15 groundstation1)
	(calibration_target instrument15 groundstation4)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star5)
	(supports instrument16 spectrograph0)
	(supports instrument16 thermograph3)
	(calibration_target instrument16 groundstation1)
	(calibration_target instrument16 star3)
	(supports instrument17 thermograph3)
	(calibration_target instrument17 groundstation1)
	(calibration_target instrument17 star3)
	(supports instrument18 spectrograph0)
	(calibration_target instrument18 star3)
	(on_board instrument16 satellite4)
	(on_board instrument17 satellite4)
	(on_board instrument18 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star6)
)
(:goal (and
	(pointing satellite4 planet7)
	(have_image star6 spectrograph2)
	(have_image planet7 spectrograph2)
	(have_image planet8 spectrograph0)
))

)
