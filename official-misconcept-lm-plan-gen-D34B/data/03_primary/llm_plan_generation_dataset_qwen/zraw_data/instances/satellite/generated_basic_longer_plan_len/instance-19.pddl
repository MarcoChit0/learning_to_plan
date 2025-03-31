(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite4 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	infrared0 - mode
	infrared1 - mode
	spectrograph3 - mode
	infrared2 - mode
	star0 - direction
	groundstation4 - direction
	star2 - direction
	groundstation3 - direction
	groundstation1 - direction
	planet5 - direction
	star6 - direction
	planet7 - direction
	planet8 - direction
	star9 - direction
	planet10 - direction
)
(:init
	(supports instrument0 infrared0)
	(supports instrument0 spectrograph3)
	(supports instrument0 infrared2)
	(calibration_target instrument0 star0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet8)
	(supports instrument2 infrared0)
	(supports instrument2 infrared1)
	(supports instrument2 infrared2)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared0)
	(supports instrument3 infrared2)
	(supports instrument3 spectrograph3)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 spectrograph3)
	(calibration_target instrument4 groundstation1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument5 infrared2)
	(supports instrument5 infrared0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 infrared2)
	(supports instrument7 infrared0)
	(supports instrument7 spectrograph3)
	(calibration_target instrument7 groundstation3)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star0)
	(supports instrument8 spectrograph3)
	(supports instrument8 infrared2)
	(calibration_target instrument8 groundstation3)
	(supports instrument9 infrared0)
	(supports instrument9 infrared1)
	(calibration_target instrument9 groundstation4)
	(supports instrument10 infrared1)
	(supports instrument10 infrared2)
	(calibration_target instrument10 star2)
	(supports instrument11 infrared2)
	(calibration_target instrument11 star2)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star6)
	(supports instrument12 spectrograph3)
	(supports instrument12 infrared2)
	(supports instrument12 infrared1)
	(calibration_target instrument12 star2)
	(supports instrument13 infrared0)
	(supports instrument13 spectrograph3)
	(calibration_target instrument13 groundstation3)
	(supports instrument14 infrared1)
	(supports instrument14 spectrograph3)
	(calibration_target instrument14 groundstation1)
	(supports instrument15 spectrograph3)
	(calibration_target instrument15 groundstation1)
	(on_board instrument12 satellite4)
	(on_board instrument13 satellite4)
	(on_board instrument14 satellite4)
	(on_board instrument15 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet10)
)
(:goal (and
	(pointing satellite0 star6)
	(pointing satellite2 planet7)
	(pointing satellite3 planet5)
	(pointing satellite4 planet8)
	(have_image planet5 infrared2)
	(have_image star6 spectrograph3)
	(have_image planet7 infrared1)
	(have_image planet8 spectrograph3)
	(have_image star9 infrared0)
	(have_image planet10 infrared0)
))

)
