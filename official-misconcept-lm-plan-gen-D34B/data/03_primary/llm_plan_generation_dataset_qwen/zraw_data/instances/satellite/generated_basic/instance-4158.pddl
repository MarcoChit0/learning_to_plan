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
	instrument14 - instrument
	thermograph3 - mode
	spectrograph1 - mode
	infrared0 - mode
	infrared2 - mode
	groundstation1 - direction
	groundstation0 - direction
	phenomenon2 - direction
)
(:init
	(supports instrument0 thermograph3)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 infrared2)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 infrared2)
	(supports instrument3 spectrograph1)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 infrared0)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon2)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument6 infrared2)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 infrared2)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 infrared0)
	(supports instrument8 thermograph3)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 groundstation0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument9 infrared0)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 spectrograph1)
	(supports instrument10 infrared2)
	(supports instrument10 thermograph3)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 infrared2)
	(supports instrument11 thermograph3)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 infrared0)
	(supports instrument12 thermograph3)
	(calibration_target instrument12 groundstation0)
	(supports instrument13 thermograph3)
	(supports instrument13 spectrograph1)
	(calibration_target instrument13 groundstation1)
	(supports instrument14 infrared0)
	(supports instrument14 spectrograph1)
	(calibration_target instrument14 groundstation0)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite2 phenomenon2)
	(have_image phenomenon2 thermograph3)
))

)
