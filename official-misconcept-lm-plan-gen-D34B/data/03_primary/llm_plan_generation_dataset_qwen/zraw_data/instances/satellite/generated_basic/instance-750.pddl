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
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	spectrograph1 - mode
	spectrograph2 - mode
	thermograph0 - mode
	groundstation5 - direction
	groundstation0 - direction
	groundstation3 - direction
	star2 - direction
	groundstation1 - direction
	star4 - direction
	star6 - direction
	planet7 - direction
	star8 - direction
	star9 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 thermograph0)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation3)
	(calibration_target instrument0 star2)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation0)
	(calibration_target instrument1 star4)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 spectrograph2)
	(supports instrument3 thermograph0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star2)
	(calibration_target instrument3 groundstation5)
	(supports instrument4 thermograph0)
	(supports instrument4 spectrograph2)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation0)
	(calibration_target instrument4 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star9)
	(supports instrument5 thermograph0)
	(supports instrument5 spectrograph2)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation1)
	(calibration_target instrument5 groundstation5)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 star4)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument7 spectrograph1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation5)
	(calibration_target instrument7 star4)
	(supports instrument8 spectrograph2)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 star4)
	(supports instrument9 spectrograph2)
	(supports instrument9 thermograph0)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 groundstation3)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 spectrograph2)
	(supports instrument10 thermograph0)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 star2)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 groundstation3)
	(calibration_target instrument11 groundstation1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet7)
	(supports instrument12 thermograph0)
	(supports instrument12 spectrograph2)
	(supports instrument12 spectrograph1)
	(calibration_target instrument12 star2)
	(supports instrument13 thermograph0)
	(supports instrument13 spectrograph1)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 thermograph0)
	(supports instrument14 spectrograph2)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 thermograph0)
	(supports instrument15 spectrograph2)
	(calibration_target instrument15 star4)
	(calibration_target instrument15 groundstation1)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet7)
)
(:goal (and
	(pointing satellite0 planet7)
	(pointing satellite2 groundstation0)
	(pointing satellite3 star9)
	(have_image star6 thermograph0)
	(have_image planet7 thermograph0)
	(have_image star8 spectrograph2)
	(have_image star9 thermograph0)
))

)
