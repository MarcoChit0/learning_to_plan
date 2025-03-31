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
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	spectrograph0 - mode
	groundstation4 - direction
	groundstation5 - direction
	star2 - direction
	groundstation3 - direction
	groundstation1 - direction
	star0 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation5)
	(calibration_target instrument3 groundstation4)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation3)
	(calibration_target instrument4 groundstation5)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation5)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation4)
	(calibration_target instrument6 star0)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation1)
	(calibration_target instrument7 groundstation5)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation3)
	(calibration_target instrument8 groundstation1)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 star2)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 star0)
	(calibration_target instrument10 star2)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation3)
	(calibration_target instrument11 groundstation1)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 star0)
	(calibration_target instrument12 groundstation1)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation4)
)
(:goal (and
	(have_image phenomenon6 spectrograph0)
))

)
