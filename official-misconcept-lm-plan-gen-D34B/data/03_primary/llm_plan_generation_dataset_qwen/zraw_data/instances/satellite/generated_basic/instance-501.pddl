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
	satellite2 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite3 - satellite
	instrument11 - instrument
	instrument12 - instrument
	spectrograph3 - mode
	infrared0 - mode
	infrared2 - mode
	infrared1 - mode
	groundstation0 - direction
	star4 - direction
	groundstation1 - direction
	star3 - direction
	groundstation2 - direction
	planet5 - direction
	phenomenon6 - direction
	phenomenon7 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 infrared0)
	(calibration_target instrument0 star3)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star3)
	(supports instrument2 infrared1)
	(supports instrument2 infrared0)
	(supports instrument2 infrared2)
	(calibration_target instrument2 star4)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star3)
	(supports instrument4 infrared2)
	(calibration_target instrument4 star4)
	(supports instrument5 infrared1)
	(supports instrument5 infrared0)
	(supports instrument5 spectrograph3)
	(calibration_target instrument5 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument6 infrared1)
	(supports instrument6 infrared2)
	(calibration_target instrument6 groundstation1)
	(supports instrument7 spectrograph3)
	(supports instrument7 infrared0)
	(calibration_target instrument7 groundstation1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon7)
	(supports instrument8 infrared1)
	(calibration_target instrument8 star3)
	(supports instrument9 infrared0)
	(supports instrument9 spectrograph3)
	(calibration_target instrument9 groundstation2)
	(supports instrument10 infrared0)
	(calibration_target instrument10 star3)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation1)
	(supports instrument11 infrared0)
	(supports instrument11 infrared2)
	(calibration_target instrument11 groundstation2)
	(supports instrument12 infrared1)
	(supports instrument12 infrared0)
	(calibration_target instrument12 groundstation2)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
)
(:goal (and
	(pointing satellite0 star3)
	(have_image planet5 infrared0)
	(have_image phenomenon6 infrared2)
	(have_image phenomenon7 infrared0)
))

)
