(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	image1 - mode
	spectrograph0 - mode
	spectrograph2 - mode
	star0 - direction
	star1 - direction
	star3 - direction
	groundstation2 - direction
	groundstation4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 spectrograph0)
	(supports instrument1 spectrograph2)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument2 spectrograph2)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation4)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
)
(:goal (and
	(pointing satellite0 star0)
	(pointing satellite1 planet5)
	(have_image planet5 spectrograph0)
))

)
