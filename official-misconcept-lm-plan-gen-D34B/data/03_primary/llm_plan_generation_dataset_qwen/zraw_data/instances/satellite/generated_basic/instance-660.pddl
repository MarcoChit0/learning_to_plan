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
	instrument7 - instrument
	spectrograph1 - mode
	spectrograph0 - mode
	star1 - direction
	star0 - direction
	groundstation3 - direction
	star2 - direction
	groundstation5 - direction
	groundstation4 - direction
	planet6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph0)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation3)
	(calibration_target instrument1 star2)
	(supports instrument2 spectrograph1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation3)
	(calibration_target instrument2 groundstation5)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet7)
	(supports instrument4 spectrograph0)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star2)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 groundstation5)
	(calibration_target instrument5 star2)
	(supports instrument6 spectrograph1)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation5)
	(supports instrument7 spectrograph1)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 groundstation4)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(on_board instrument7 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(pointing satellite0 star0)
	(have_image planet6 spectrograph0)
	(have_image planet7 spectrograph1)
	(have_image planet8 spectrograph0)
))

)
