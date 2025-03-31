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
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	satellite3 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	infrared3 - mode
	infrared1 - mode
	spectrograph2 - mode
	image0 - mode
	star2 - direction
	star3 - direction
	groundstation0 - direction
	groundstation1 - direction
	star4 - direction
	phenomenon5 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 infrared1)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared3)
	(supports instrument1 image0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 star2)
	(supports instrument2 image0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument3 image0)
	(supports instrument3 infrared1)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 infrared1)
	(supports instrument4 spectrograph2)
	(supports instrument4 image0)
	(calibration_target instrument4 star2)
	(supports instrument5 infrared1)
	(supports instrument5 image0)
	(supports instrument5 infrared3)
	(calibration_target instrument5 groundstation1)
	(supports instrument6 infrared1)
	(calibration_target instrument6 star3)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument7 image0)
	(supports instrument7 infrared3)
	(calibration_target instrument7 star3)
	(supports instrument8 infrared3)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 spectrograph2)
	(supports instrument9 infrared3)
	(calibration_target instrument9 groundstation1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
	(supports instrument10 infrared1)
	(supports instrument10 infrared3)
	(supports instrument10 spectrograph2)
	(calibration_target instrument10 groundstation1)
	(supports instrument11 spectrograph2)
	(calibration_target instrument11 star4)
	(supports instrument12 infrared1)
	(calibration_target instrument12 star4)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation0)
)
(:goal (and
	(pointing satellite0 star4)
	(have_image phenomenon5 spectrograph2)
	(have_image star6 infrared1)
))

)
