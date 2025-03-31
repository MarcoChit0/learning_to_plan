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
	satellite2 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	satellite3 - satellite
	instrument16 - instrument
	spectrograph3 - mode
	image1 - mode
	thermograph2 - mode
	spectrograph0 - mode
	star4 - direction
	groundstation3 - direction
	groundstation0 - direction
	groundstation1 - direction
	groundstation2 - direction
	planet5 - direction
	phenomenon6 - direction
	planet7 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 spectrograph3)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 spectrograph0)
	(supports instrument1 spectrograph3)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph3)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 spectrograph0)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 spectrograph0)
	(supports instrument7 spectrograph3)
	(calibration_target instrument7 groundstation3)
	(supports instrument8 spectrograph0)
	(supports instrument8 thermograph2)
	(supports instrument8 spectrograph3)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 spectrograph0)
	(supports instrument9 image1)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 groundstation0)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument10 spectrograph3)
	(calibration_target instrument10 groundstation3)
	(supports instrument11 spectrograph0)
	(supports instrument11 spectrograph3)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 thermograph2)
	(calibration_target instrument12 groundstation2)
	(supports instrument13 spectrograph3)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 image1)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 image1)
	(supports instrument15 thermograph2)
	(calibration_target instrument15 groundstation1)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(on_board instrument14 satellite2)
	(on_board instrument15 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument16 thermograph2)
	(supports instrument16 spectrograph3)
	(supports instrument16 image1)
	(calibration_target instrument16 groundstation2)
	(on_board instrument16 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite3 groundstation1)
	(have_image planet5 image1)
	(have_image phenomenon6 thermograph2)
	(have_image planet7 image1)
))

)
