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
	infrared0 - mode
	infrared1 - mode
	star0 - direction
	groundstation4 - direction
	groundstation2 - direction
	groundstation3 - direction
	groundstation1 - direction
	planet5 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation3)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument2 infrared0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation3)
	(supports instrument3 infrared0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation1)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
)
(:goal (and
	(pointing satellite0 groundstation2)
	(pointing satellite1 groundstation3)
	(have_image planet5 infrared0)
))

)
