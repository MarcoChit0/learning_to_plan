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
	satellite2 - satellite
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	spectrograph0 - mode
	star0 - direction
	groundstation3 - direction
	groundstation2 - direction
	groundstation5 - direction
	star4 - direction
	star1 - direction
	star6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation5)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star4)
	(calibration_target instrument1 groundstation3)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star4)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star4)
	(calibration_target instrument5 star1)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation5)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation5)
	(calibration_target instrument6 groundstation2)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star4)
	(calibration_target instrument7 groundstation5)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 star1)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(on_board instrument8 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation5)
)
(:goal (and
	(pointing satellite2 groundstation2)
	(pointing satellite3 star4)
	(have_image star6 spectrograph0)
))

)
