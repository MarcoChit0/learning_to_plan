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
	satellite2 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite3 - satellite
	instrument14 - instrument
	instrument15 - instrument
	instrument16 - instrument
	instrument17 - instrument
	infrared3 - mode
	spectrograph2 - mode
	spectrograph0 - mode
	thermograph1 - mode
	groundstation2 - direction
	star0 - direction
	groundstation3 - direction
	groundstation1 - direction
	star4 - direction
	star5 - direction
	planet6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 infrared3)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 infrared3)
	(supports instrument3 spectrograph0)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 spectrograph2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon7)
	(supports instrument5 infrared3)
	(supports instrument5 thermograph1)
	(supports instrument5 spectrograph2)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 spectrograph2)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 spectrograph0)
	(supports instrument7 spectrograph2)
	(supports instrument7 infrared3)
	(calibration_target instrument7 star0)
	(supports instrument8 thermograph1)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 spectrograph0)
	(supports instrument9 spectrograph2)
	(calibration_target instrument9 groundstation1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument10 thermograph1)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 thermograph1)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation3)
	(supports instrument12 spectrograph0)
	(supports instrument12 spectrograph2)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 star4)
	(supports instrument13 spectrograph2)
	(supports instrument13 thermograph1)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 star4)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(on_board instrument12 satellite2)
	(on_board instrument13 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
	(supports instrument14 spectrograph0)
	(supports instrument14 thermograph1)
	(supports instrument14 spectrograph2)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 infrared3)
	(supports instrument15 spectrograph0)
	(calibration_target instrument15 star4)
	(supports instrument16 infrared3)
	(supports instrument16 spectrograph2)
	(supports instrument16 spectrograph0)
	(calibration_target instrument16 groundstation1)
	(supports instrument17 spectrograph0)
	(supports instrument17 spectrograph2)
	(calibration_target instrument17 star4)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(on_board instrument16 satellite3)
	(on_board instrument17 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite1 star5)
	(pointing satellite2 groundstation2)
	(have_image star5 spectrograph0)
	(have_image planet6 spectrograph2)
	(have_image phenomenon7 thermograph1)
))

)
