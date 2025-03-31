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
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	spectrograph0 - mode
	groundstation0 - direction
	groundstation5 - direction
	groundstation1 - direction
	star3 - direction
	star4 - direction
	star2 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation5)
	(calibration_target instrument0 star3)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star4)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation5)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star3)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star2)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation5)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star3)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star4)
	(calibration_target instrument6 star3)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation1)
	(calibration_target instrument7 star2)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star4)
	(calibration_target instrument8 star3)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star2)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation1)
)
(:goal (and
	(pointing satellite1 star4)
	(pointing satellite3 phenomenon6)
	(have_image phenomenon6 spectrograph0)
))

)
