(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	spectrograph2 - mode
	infrared1 - mode
	thermograph3 - mode
	image0 - mode
	star0 - direction
	star1 - direction
	star2 - direction
	groundstation4 - direction
	star3 - direction
	planet5 - direction
	phenomenon6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 infrared1)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star3)
	(supports instrument2 spectrograph2)
	(supports instrument2 image0)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 star3)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation4)
)
(:goal (and
	(pointing satellite0 phenomenon6)
	(have_image planet5 infrared1)
	(have_image phenomenon6 spectrograph2)
	(have_image planet7 spectrograph2)
	(have_image phenomenon8 thermograph3)
))

)
