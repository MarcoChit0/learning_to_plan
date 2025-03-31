(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	spectrograph0 - mode
	image1 - mode
	groundstation3 - direction
	star4 - direction
	groundstation2 - direction
	groundstation0 - direction
	star1 - direction
	phenomenon5 - direction
	planet6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 spectrograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon8)
	(supports instrument2 spectrograph0)
	(supports instrument2 image1)
	(calibration_target instrument2 star1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet6)
)
(:goal (and
	(pointing satellite1 groundstation0)
	(have_image phenomenon5 spectrograph0)
	(have_image planet6 spectrograph0)
	(have_image planet7 spectrograph0)
	(have_image phenomenon8 spectrograph0)
))

)
