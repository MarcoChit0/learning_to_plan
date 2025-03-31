(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	satellite4 - satellite
	instrument9 - instrument
	image0 - mode
	spectrograph1 - mode
	groundstation3 - direction
	groundstation4 - direction
	groundstation0 - direction
	groundstation1 - direction
	star5 - direction
	star2 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
	star9 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation4)
	(calibration_target instrument0 groundstation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation0)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation1)
	(calibration_target instrument3 star2)
	(supports instrument4 image0)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star2)
	(calibration_target instrument4 star5)
	(supports instrument5 image0)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star5)
	(calibration_target instrument5 groundstation4)
	(supports instrument6 image0)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 star5)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star9)
	(supports instrument7 image0)
	(supports instrument7 spectrograph1)
	(calibration_target instrument7 star2)
	(calibration_target instrument7 groundstation0)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument8 spectrograph1)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation1)
	(calibration_target instrument8 star5)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation3)
	(supports instrument9 spectrograph1)
	(supports instrument9 image0)
	(calibration_target instrument9 star2)
	(calibration_target instrument9 star5)
	(on_board instrument9 satellite4)
	(power_avail satellite4)
	(pointing satellite4 star5)
)
(:goal (and
	(pointing satellite1 star5)
	(pointing satellite2 groundstation3)
	(have_image planet6 spectrograph1)
	(have_image star7 spectrograph1)
	(have_image planet8 spectrograph1)
	(have_image star9 image0)
))

)
