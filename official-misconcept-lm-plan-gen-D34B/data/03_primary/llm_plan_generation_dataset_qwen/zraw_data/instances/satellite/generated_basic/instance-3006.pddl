(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	spectrograph0 - mode
	groundstation1 - direction
	groundstation3 - direction
	groundstation2 - direction
	groundstation5 - direction
	groundstation0 - direction
	star4 - direction
	phenomenon6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation3)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation3)
	(calibration_target instrument2 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star4)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation0)
	(calibration_target instrument3 star4)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star4)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation5)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation0)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 star4)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet8)
)
(:goal (and
	(pointing satellite1 star4)
	(pointing satellite3 planet7)
	(have_image phenomenon6 spectrograph0)
	(have_image planet7 spectrograph0)
	(have_image planet8 spectrograph0)
))

)
