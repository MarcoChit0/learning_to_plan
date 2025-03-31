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
	spectrograph1 - mode
	spectrograph2 - mode
	spectrograph0 - mode
	groundstation2 - direction
	groundstation0 - direction
	groundstation3 - direction
	star1 - direction
	planet4 - direction
	star5 - direction
	planet6 - direction
	planet7 - direction
	planet8 - direction
	star9 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation0)
	(supports instrument1 spectrograph1)
	(supports instrument1 spectrograph2)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star5)
	(supports instrument2 spectrograph2)
	(supports instrument2 spectrograph0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation3)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
	(supports instrument3 spectrograph2)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star1)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star1)
)
(:goal (and
	(pointing satellite0 star1)
	(pointing satellite1 star1)
	(have_image planet4 spectrograph1)
	(have_image star5 spectrograph2)
	(have_image planet6 spectrograph0)
	(have_image planet7 spectrograph0)
	(have_image planet8 spectrograph0)
	(have_image star9 spectrograph2)
))

)
