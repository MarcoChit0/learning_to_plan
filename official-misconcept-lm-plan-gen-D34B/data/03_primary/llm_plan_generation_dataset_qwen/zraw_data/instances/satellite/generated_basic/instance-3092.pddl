(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	spectrograph0 - mode
	groundstation3 - direction
	groundstation0 - direction
	star2 - direction
	groundstation1 - direction
	planet4 - direction
	planet5 - direction
	star6 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet4)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star2)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 groundstation1)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
)
(:goal (and
	(have_image planet4 spectrograph0)
	(have_image planet5 spectrograph0)
	(have_image star6 spectrograph0)
))

)
