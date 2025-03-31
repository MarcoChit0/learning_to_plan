(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
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
	spectrograph2 - mode
	spectrograph3 - mode
	infrared1 - mode
	image0 - mode
	groundstation0 - direction
	star1 - direction
	planet2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 infrared1)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet3)
	(supports instrument4 infrared1)
	(supports instrument4 spectrograph3)
	(supports instrument4 spectrograph2)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 infrared1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph2)
	(supports instrument6 image0)
	(supports instrument6 spectrograph3)
	(calibration_target instrument6 groundstation0)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet3)
	(supports instrument7 image0)
	(supports instrument7 spectrograph2)
	(supports instrument7 infrared1)
	(calibration_target instrument7 groundstation0)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation0)
	(supports instrument9 spectrograph2)
	(supports instrument9 image0)
	(supports instrument9 spectrograph3)
	(calibration_target instrument9 groundstation0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet2)
	(supports instrument10 infrared1)
	(supports instrument10 image0)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 spectrograph2)
	(supports instrument11 spectrograph3)
	(calibration_target instrument11 groundstation0)
	(supports instrument12 spectrograph3)
	(calibration_target instrument12 groundstation0)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
)
(:goal (and
	(pointing satellite1 planet3)
	(have_image star1 infrared1)
	(have_image planet2 spectrograph3)
	(have_image planet3 spectrograph3)
))

)
