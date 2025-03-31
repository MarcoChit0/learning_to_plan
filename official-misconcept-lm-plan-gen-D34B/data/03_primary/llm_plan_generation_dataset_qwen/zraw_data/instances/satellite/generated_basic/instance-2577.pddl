(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	image0 - mode
	spectrograph2 - mode
	infrared1 - mode
	groundstation0 - direction
	groundstation3 - direction
	groundstation1 - direction
	star4 - direction
	groundstation2 - direction
	planet5 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 image0)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument2 infrared1)
	(supports instrument2 image0)
	(calibration_target instrument2 star4)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
)
(:goal (and
	(have_image planet5 infrared1)
))

)
