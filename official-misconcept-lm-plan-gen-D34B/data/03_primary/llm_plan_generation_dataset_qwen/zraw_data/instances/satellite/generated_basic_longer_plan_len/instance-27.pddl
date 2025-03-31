(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	image1 - mode
	spectrograph0 - mode
	infrared2 - mode
	infrared3 - mode
	groundstation2 - direction
	groundstation1 - direction
	groundstation0 - direction
	groundstation3 - direction
	star4 - direction
	phenomenon5 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
	star9 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared3)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation2)
	(supports instrument2 spectrograph0)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument4 infrared3)
	(supports instrument4 infrared2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 spectrograph0)
	(supports instrument5 infrared3)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 infrared3)
	(supports instrument6 image1)
	(supports instrument6 infrared2)
	(calibration_target instrument6 groundstation3)
	(supports instrument7 image1)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 infrared3)
	(calibration_target instrument8 groundstation1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star7)
	(supports instrument9 infrared3)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation0)
	(supports instrument10 image1)
	(supports instrument10 spectrograph0)
	(supports instrument10 infrared2)
	(calibration_target instrument10 groundstation3)
	(supports instrument11 infrared2)
	(supports instrument11 image1)
	(calibration_target instrument11 groundstation3)
	(supports instrument12 spectrograph0)
	(supports instrument12 infrared2)
	(calibration_target instrument12 groundstation3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star9)
)
(:goal (and
	(pointing satellite0 groundstation3)
	(have_image star4 image1)
	(have_image phenomenon5 infrared2)
	(have_image star6 infrared3)
	(have_image star7 infrared3)
	(have_image phenomenon8 spectrograph0)
	(have_image star9 infrared2)
))

)
