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
	infrared1 - mode
	spectrograph2 - mode
	groundstation1 - direction
	star3 - direction
	star0 - direction
	star2 - direction
	star4 - direction
	planet5 - direction
	star6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 image0)
	(calibration_target instrument0 star3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument1 infrared1)
	(supports instrument1 spectrograph2)
	(supports instrument1 image0)
	(calibration_target instrument1 star2)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
	(supports instrument2 spectrograph2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared1)
	(supports instrument3 image0)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star2)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon8)
)
(:goal (and
	(pointing satellite0 planet7)
	(have_image star4 infrared1)
	(have_image planet5 spectrograph2)
	(have_image star6 spectrograph2)
	(have_image planet7 infrared1)
	(have_image phenomenon8 infrared1)
))

)
