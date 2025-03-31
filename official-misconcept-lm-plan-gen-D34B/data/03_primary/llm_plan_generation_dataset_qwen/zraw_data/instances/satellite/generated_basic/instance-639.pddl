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
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite4 - satellite
	instrument14 - instrument
	instrument15 - instrument
	thermograph2 - mode
	spectrograph1 - mode
	infrared0 - mode
	groundstation0 - direction
	star3 - direction
	star1 - direction
	groundstation2 - direction
	star4 - direction
	star5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 thermograph2)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 thermograph2)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 spectrograph1)
	(supports instrument3 infrared0)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star1)
	(supports instrument4 infrared0)
	(supports instrument4 thermograph2)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument5 thermograph2)
	(supports instrument5 infrared0)
	(calibration_target instrument5 star1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 thermograph2)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 groundstation0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 star3)
	(supports instrument10 spectrograph1)
	(supports instrument10 thermograph2)
	(supports instrument10 infrared0)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 spectrograph1)
	(supports instrument11 thermograph2)
	(calibration_target instrument11 star3)
	(supports instrument12 spectrograph1)
	(supports instrument12 infrared0)
	(calibration_target instrument12 star3)
	(supports instrument13 spectrograph1)
	(supports instrument13 thermograph2)
	(supports instrument13 infrared0)
	(calibration_target instrument13 groundstation2)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star3)
	(supports instrument14 infrared0)
	(supports instrument14 thermograph2)
	(calibration_target instrument14 star1)
	(supports instrument15 spectrograph1)
	(calibration_target instrument15 groundstation2)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet6)
)
(:goal (and
	(have_image star4 infrared0)
	(have_image star5 spectrograph1)
	(have_image planet6 thermograph2)
))

)
