(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	image0 - mode
	image3 - mode
	spectrograph1 - mode
	spectrograph2 - mode
	groundstation1 - direction
	star4 - direction
	star3 - direction
	star2 - direction
	groundstation0 - direction
	phenomenon5 - direction
	planet6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star3)
	(supports instrument1 spectrograph1)
	(supports instrument1 spectrograph2)
	(supports instrument1 image3)
	(calibration_target instrument1 star2)
	(supports instrument2 image0)
	(supports instrument2 image3)
	(supports instrument2 spectrograph1)
	(calibration_target instrument2 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet7)
)
(:goal (and
	(pointing satellite1 phenomenon5)
	(have_image phenomenon5 image3)
	(have_image planet6 spectrograph1)
	(have_image planet7 spectrograph1)
	(have_image star8 image0)
))

)
