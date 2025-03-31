(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	image1 - mode
	spectrograph0 - mode
	star3 - direction
	star5 - direction
	groundstation2 - direction
	star0 - direction
	star4 - direction
	groundstation1 - direction
	planet6 - direction
	star7 - direction
	star8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation2)
	(calibration_target instrument0 star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument1 spectrograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 star0)
	(supports instrument2 spectrograph0)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation1)
	(calibration_target instrument2 star4)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
)
(:goal (and
	(pointing satellite0 star3)
	(pointing satellite1 star7)
	(have_image planet6 image1)
	(have_image star7 spectrograph0)
	(have_image star8 image1)
))

)
