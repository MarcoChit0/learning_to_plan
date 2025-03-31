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
	image2 - mode
	star0 - direction
	groundstation3 - direction
	groundstation1 - direction
	star2 - direction
	star4 - direction
	planet5 - direction
	planet6 - direction
	star7 - direction
	planet8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 image2)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 image1)
	(supports instrument1 image2)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 image2)
	(calibration_target instrument2 star4)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
)
(:goal (and
	(have_image planet5 image2)
	(have_image planet6 image1)
	(have_image star7 spectrograph0)
	(have_image planet8 image1)
	(have_image phenomenon9 image2)
))

)
