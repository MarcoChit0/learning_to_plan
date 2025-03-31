(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	infrared2 - mode
	spectrograph0 - mode
	image3 - mode
	thermograph1 - mode
	star1 - direction
	groundstation4 - direction
	groundstation2 - direction
	groundstation3 - direction
	groundstation0 - direction
	planet5 - direction
	star6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 image3)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 image3)
	(calibration_target instrument2 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet8)
	(supports instrument3 image3)
	(supports instrument3 thermograph1)
	(supports instrument3 infrared2)
	(calibration_target instrument3 groundstation0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
)
(:goal (and
	(have_image planet5 thermograph1)
	(have_image star6 image3)
	(have_image star7 image3)
	(have_image planet8 infrared2)
))

)
