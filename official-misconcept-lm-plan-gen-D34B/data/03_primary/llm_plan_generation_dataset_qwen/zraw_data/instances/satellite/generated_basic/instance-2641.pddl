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
	satellite3 - satellite
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	spectrograph0 - mode
	star2 - direction
	star3 - direction
	groundstation4 - direction
	star5 - direction
	star0 - direction
	groundstation1 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star2)
	(calibration_target instrument0 star3)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star2)
	(calibration_target instrument1 star3)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star3)
	(calibration_target instrument3 star5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star3)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star3)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star5)
	(calibration_target instrument7 groundstation1)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation1)
	(calibration_target instrument8 star5)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon6)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation4)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 star5)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation1)
	(calibration_target instrument11 star0)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star0)
)
(:goal (and
	(pointing satellite0 star2)
	(pointing satellite1 star3)
	(pointing satellite3 star3)
	(have_image phenomenon6 spectrograph0)
))

)
