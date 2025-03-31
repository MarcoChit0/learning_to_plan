(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	spectrograph2 - mode
	infrared1 - mode
	image3 - mode
	thermograph0 - mode
	groundstation0 - direction
	planet1 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 image3)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument1 thermograph0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 spectrograph2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 spectrograph2)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation0)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet1)
)
(:goal (and
	(pointing satellite0 planet1)
	(have_image planet1 thermograph0)
))

)
