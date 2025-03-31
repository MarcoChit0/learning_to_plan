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
	satellite3 - satellite
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	thermograph2 - mode
	infrared0 - mode
	image3 - mode
	spectrograph1 - mode
	star0 - direction
	groundstation1 - direction
	phenomenon2 - direction
	planet3 - direction
	star4 - direction
	planet5 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image3)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 thermograph2)
	(supports instrument1 spectrograph1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 image3)
	(calibration_target instrument3 star0)
	(supports instrument4 image3)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 infrared0)
	(supports instrument5 thermograph2)
	(calibration_target instrument5 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument6 infrared0)
	(supports instrument6 image3)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 star0)
	(supports instrument7 image3)
	(supports instrument7 thermograph2)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 thermograph2)
	(supports instrument8 spectrograph1)
	(supports instrument8 infrared0)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 groundstation1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet5)
	(supports instrument10 spectrograph1)
	(calibration_target instrument10 groundstation1)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon2)
	(supports instrument11 image3)
	(supports instrument11 thermograph2)
	(supports instrument11 infrared0)
	(calibration_target instrument11 star0)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon2)
	(supports instrument12 spectrograph1)
	(supports instrument12 image3)
	(calibration_target instrument12 groundstation1)
	(on_board instrument12 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon6)
)
(:goal (and
	(pointing satellite0 groundstation1)
	(pointing satellite1 groundstation1)
	(pointing satellite3 planet5)
	(have_image phenomenon2 spectrograph1)
	(have_image planet3 infrared0)
	(have_image star4 spectrograph1)
	(have_image planet5 thermograph2)
	(have_image phenomenon6 thermograph2)
))

)
