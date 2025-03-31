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
	satellite2 - satellite
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite4 - satellite
	instrument11 - instrument
	satellite5 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	infrared3 - mode
	thermograph1 - mode
	spectrograph0 - mode
	image2 - mode
	groundstation2 - direction
	star0 - direction
	groundstation1 - direction
	star3 - direction
	phenomenon4 - direction
	planet5 - direction
	planet6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 infrared3)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 spectrograph0)
	(supports instrument1 image2)
	(supports instrument1 infrared3)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 image2)
	(calibration_target instrument2 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument3 infrared3)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star0)
	(supports instrument4 spectrograph0)
	(supports instrument4 image2)
	(calibration_target instrument4 groundstation2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument5 image2)
	(calibration_target instrument5 groundstation1)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument6 thermograph1)
	(supports instrument6 infrared3)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 infrared3)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 spectrograph0)
	(supports instrument8 image2)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared3)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 groundstation1)
	(supports instrument10 image2)
	(supports instrument10 thermograph1)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 star0)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon4)
	(supports instrument11 infrared3)
	(calibration_target instrument11 star0)
	(on_board instrument11 satellite4)
	(power_avail satellite4)
	(pointing satellite4 groundstation2)
	(supports instrument12 image2)
	(supports instrument12 thermograph1)
	(calibration_target instrument12 star0)
	(supports instrument13 infrared3)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 infrared3)
	(calibration_target instrument14 groundstation1)
	(on_board instrument12 satellite5)
	(on_board instrument13 satellite5)
	(on_board instrument14 satellite5)
	(power_avail satellite5)
	(pointing satellite5 star0)
)
(:goal (and
	(pointing satellite5 planet6)
	(have_image star3 image2)
	(have_image phenomenon4 infrared3)
	(have_image planet5 spectrograph0)
	(have_image planet6 image2)
	(have_image phenomenon7 image2)
))

)
