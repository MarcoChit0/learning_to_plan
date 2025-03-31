(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	spectrograph0 - mode
	groundstation0 - direction
	groundstation1 - direction
	star2 - direction
	groundstation4 - direction
	groundstation3 - direction
	phenomenon5 - direction
	star6 - direction
	star7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star2)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star2)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star2)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation4)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation4)
)
(:goal (and
	(pointing satellite0 groundstation4)
	(have_image phenomenon5 spectrograph0)
	(have_image star6 spectrograph0)
	(have_image star7 spectrograph0)
	(have_image phenomenon8 spectrograph0)
))

)
