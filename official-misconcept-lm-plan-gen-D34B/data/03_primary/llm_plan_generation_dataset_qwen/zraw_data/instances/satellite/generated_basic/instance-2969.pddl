(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	satellite4 - satellite
	instrument8 - instrument
	instrument9 - instrument
	thermograph3 - mode
	infrared0 - mode
	infrared2 - mode
	spectrograph1 - mode
	groundstation0 - direction
	star1 - direction
	star2 - direction
	groundstation3 - direction
	planet4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 infrared2)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument1 thermograph3)
	(supports instrument1 infrared0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 star2)
	(supports instrument3 thermograph3)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 infrared2)
	(calibration_target instrument5 star1)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 star2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument7 spectrograph1)
	(supports instrument7 infrared2)
	(supports instrument7 thermograph3)
	(calibration_target instrument7 star1)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
	(supports instrument8 spectrograph1)
	(calibration_target instrument8 star2)
	(supports instrument9 infrared0)
	(supports instrument9 spectrograph1)
	(calibration_target instrument9 groundstation3)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star1)
)
(:goal (and
	(pointing satellite2 star1)
	(have_image planet4 infrared0)
	(have_image phenomenon5 spectrograph1)
))

)
