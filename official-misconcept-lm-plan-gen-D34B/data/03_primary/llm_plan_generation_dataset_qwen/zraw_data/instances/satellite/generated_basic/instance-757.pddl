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
	satellite3 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	infrared2 - mode
	spectrograph0 - mode
	image1 - mode
	star0 - direction
	star4 - direction
	star2 - direction
	groundstation5 - direction
	star1 - direction
	groundstation3 - direction
	phenomenon6 - direction
	star7 - direction
	planet8 - direction
	star9 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation3)
	(calibration_target instrument0 star4)
	(supports instrument1 infrared2)
	(supports instrument1 image1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation3)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 spectrograph0)
	(supports instrument2 image1)
	(supports instrument2 infrared2)
	(calibration_target instrument2 star4)
	(supports instrument3 image1)
	(supports instrument3 infrared2)
	(calibration_target instrument3 star0)
	(supports instrument4 infrared2)
	(supports instrument4 image1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star4)
	(calibration_target instrument4 groundstation3)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation5)
	(supports instrument6 infrared2)
	(supports instrument6 image1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star9)
	(supports instrument7 spectrograph0)
	(supports instrument7 image1)
	(calibration_target instrument7 star4)
	(supports instrument8 image1)
	(calibration_target instrument8 star2)
	(calibration_target instrument8 groundstation5)
	(supports instrument9 image1)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 infrared2)
	(supports instrument10 image1)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation5)
	(supports instrument11 spectrograph0)
	(supports instrument11 image1)
	(supports instrument11 infrared2)
	(calibration_target instrument11 groundstation3)
	(calibration_target instrument11 star1)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
)
(:goal (and
	(pointing satellite0 star1)
	(have_image phenomenon6 image1)
	(have_image star7 image1)
	(have_image planet8 image1)
	(have_image star9 image1)
))

)
