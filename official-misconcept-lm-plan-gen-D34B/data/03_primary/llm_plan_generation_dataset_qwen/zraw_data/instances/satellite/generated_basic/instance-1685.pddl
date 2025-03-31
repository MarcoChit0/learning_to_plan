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
	instrument6 - instrument
	satellite2 - satellite
	instrument7 - instrument
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	instrument13 - instrument
	instrument14 - instrument
	instrument15 - instrument
	spectrograph0 - mode
	star4 - direction
	groundstation5 - direction
	groundstation1 - direction
	star2 - direction
	groundstation3 - direction
	groundstation0 - direction
	planet6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation3)
	(calibration_target instrument0 groundstation5)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation3)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation5)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star4)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation1)
	(calibration_target instrument4 groundstation0)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 star2)
	(calibration_target instrument6 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation1)
	(calibration_target instrument7 groundstation5)
	(supports instrument8 spectrograph0)
	(calibration_target instrument8 groundstation5)
	(supports instrument9 spectrograph0)
	(calibration_target instrument9 groundstation3)
	(supports instrument10 spectrograph0)
	(calibration_target instrument10 groundstation0)
	(supports instrument11 spectrograph0)
	(calibration_target instrument11 groundstation0)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument12 spectrograph0)
	(calibration_target instrument12 groundstation3)
	(calibration_target instrument12 groundstation1)
	(supports instrument13 spectrograph0)
	(calibration_target instrument13 groundstation3)
	(supports instrument14 spectrograph0)
	(calibration_target instrument14 star2)
	(calibration_target instrument14 groundstation3)
	(supports instrument15 spectrograph0)
	(calibration_target instrument15 groundstation0)
	(calibration_target instrument15 groundstation3)
	(on_board instrument12 satellite3)
	(on_board instrument13 satellite3)
	(on_board instrument14 satellite3)
	(on_board instrument15 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet6)
)
(:goal (and
	(pointing satellite1 star2)
	(have_image planet6 spectrograph0)
))

)
