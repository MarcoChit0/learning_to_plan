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
	image0 - mode
	spectrograph2 - mode
	image1 - mode
	groundstation1 - direction
	groundstation4 - direction
	star2 - direction
	groundstation3 - direction
	groundstation5 - direction
	star0 - direction
	planet6 - direction
	planet7 - direction
	star8 - direction
	star9 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 image1)
	(supports instrument0 image0)
	(calibration_target instrument0 star2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument1 spectrograph2)
	(supports instrument1 image1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation3)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument2 image0)
	(supports instrument2 image1)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 star0)
	(supports instrument3 image1)
	(supports instrument3 image0)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star0)
	(calibration_target instrument3 groundstation5)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet6)
)
(:goal (and
	(pointing satellite0 star0)
	(have_image planet6 image1)
	(have_image planet7 image0)
	(have_image star8 image1)
	(have_image star9 image1)
))

)
