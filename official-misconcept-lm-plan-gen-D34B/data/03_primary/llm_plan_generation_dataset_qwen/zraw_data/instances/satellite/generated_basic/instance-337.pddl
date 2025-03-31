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
	spectrograph1 - mode
	spectrograph0 - mode
	star1 - direction
	star2 - direction
	star0 - direction
	groundstation3 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	planet6 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument1 spectrograph1)
	(calibration_target instrument1 groundstation3)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 star0)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation3)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star2)
)
(:goal (and
	(pointing satellite1 phenomenon5)
	(pointing satellite2 groundstation3)
	(have_image phenomenon4 spectrograph0)
	(have_image phenomenon5 spectrograph0)
	(have_image planet6 spectrograph0)
))

)
