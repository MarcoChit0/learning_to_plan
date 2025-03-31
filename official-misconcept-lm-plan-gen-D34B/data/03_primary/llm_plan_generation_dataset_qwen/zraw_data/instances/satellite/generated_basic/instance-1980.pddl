(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	spectrograph0 - mode
	groundstation3 - direction
	groundstation0 - direction
	star2 - direction
	star1 - direction
	phenomenon4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon4)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star2)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
)
(:goal (and
	(have_image phenomenon4 spectrograph0)
	(have_image phenomenon5 spectrograph0)
))

)
