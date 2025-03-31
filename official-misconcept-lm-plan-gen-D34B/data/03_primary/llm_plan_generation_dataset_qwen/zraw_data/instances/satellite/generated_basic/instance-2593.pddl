(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	spectrograph1 - mode
	image3 - mode
	image2 - mode
	image0 - mode
	star0 - direction
	star1 - direction
	star2 - direction
)
(:init
	(supports instrument0 image3)
	(supports instrument0 image0)
	(supports instrument0 image2)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument1 spectrograph1)
	(supports instrument1 image2)
	(supports instrument1 image3)
	(calibration_target instrument1 star0)
	(supports instrument2 image0)
	(supports instrument2 image3)
	(calibration_target instrument2 star1)
	(supports instrument3 image2)
	(supports instrument3 image3)
	(supports instrument3 image0)
	(calibration_target instrument3 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
)
(:goal (and
	(pointing satellite1 star1)
	(have_image star2 spectrograph1)
))

)
