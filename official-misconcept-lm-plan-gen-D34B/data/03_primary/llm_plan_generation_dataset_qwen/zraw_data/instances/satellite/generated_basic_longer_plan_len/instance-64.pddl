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
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	spectrograph3 - mode
	image0 - mode
	infrared2 - mode
	thermograph1 - mode
	star0 - direction
	star3 - direction
	star2 - direction
	groundstation1 - direction
	planet4 - direction
	planet5 - direction
	phenomenon6 - direction
	planet7 - direction
	planet8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 spectrograph3)
	(supports instrument0 thermograph1)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(calibration_target instrument1 star2)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet4)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 star0)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 star3)
	(supports instrument5 thermograph1)
	(supports instrument5 image0)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 star2)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument6 image0)
	(supports instrument6 thermograph1)
	(supports instrument6 infrared2)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 thermograph1)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 image0)
	(calibration_target instrument8 star0)
	(supports instrument9 infrared2)
	(supports instrument9 image0)
	(supports instrument9 thermograph1)
	(calibration_target instrument9 star0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet7)
	(supports instrument10 thermograph1)
	(supports instrument10 infrared2)
	(supports instrument10 image0)
	(calibration_target instrument10 star3)
	(supports instrument11 infrared2)
	(supports instrument11 image0)
	(calibration_target instrument11 star3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star2)
	(supports instrument12 infrared2)
	(calibration_target instrument12 star2)
	(supports instrument13 infrared2)
	(supports instrument13 image0)
	(supports instrument13 thermograph1)
	(calibration_target instrument13 star2)
	(supports instrument14 infrared2)
	(supports instrument14 thermograph1)
	(supports instrument14 spectrograph3)
	(calibration_target instrument14 groundstation1)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star0)
)
(:goal (and
	(pointing satellite0 planet5)
	(pointing satellite1 star0)
	(pointing satellite3 planet7)
	(have_image planet4 thermograph1)
	(have_image planet5 thermograph1)
	(have_image phenomenon6 image0)
	(have_image planet7 spectrograph3)
	(have_image planet8 image0)
	(have_image planet9 spectrograph3)
))

)
