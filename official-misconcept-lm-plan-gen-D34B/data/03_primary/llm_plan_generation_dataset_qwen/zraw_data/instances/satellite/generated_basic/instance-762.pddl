(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite2 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite3 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	infrared1 - mode
	thermograph0 - mode
	spectrograph2 - mode
	groundstation1 - direction
	star5 - direction
	groundstation2 - direction
	groundstation3 - direction
	groundstation0 - direction
	star4 - direction
	star6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
	star9 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared1)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star6)
	(supports instrument3 infrared1)
	(supports instrument3 spectrograph2)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 groundstation0)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation2)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph2)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star4)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 spectrograph2)
	(calibration_target instrument8 star4)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument9 infrared1)
	(supports instrument9 spectrograph2)
	(supports instrument9 thermograph0)
	(calibration_target instrument9 star5)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 thermograph0)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 thermograph0)
	(supports instrument11 spectrograph2)
	(calibration_target instrument11 star4)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 infrared1)
	(calibration_target instrument12 star4)
	(supports instrument13 spectrograph2)
	(calibration_target instrument13 star4)
	(calibration_target instrument13 groundstation3)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon8)
	(supports instrument14 thermograph0)
	(supports instrument14 infrared1)
	(calibration_target instrument14 groundstation0)
	(calibration_target instrument14 groundstation3)
	(supports instrument15 thermograph0)
	(calibration_target instrument15 star4)
	(calibration_target instrument15 groundstation0)
	(supports instrument16 spectrograph2)
	(supports instrument16 thermograph0)
	(supports instrument16 infrared1)
	(calibration_target instrument16 groundstation0)
	(supports instrument17 thermograph0)
	(calibration_target instrument17 star4)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon8)
)
(:goal (and
	(pointing satellite2 groundstation3)
	(pointing satellite3 phenomenon8)
	(have_image star6 thermograph0)
	(have_image phenomenon7 infrared1)
	(have_image phenomenon8 infrared1)
	(have_image star9 spectrograph2)
))

)
