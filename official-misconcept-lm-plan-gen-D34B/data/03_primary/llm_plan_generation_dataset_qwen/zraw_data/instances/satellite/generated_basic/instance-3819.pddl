(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	infrared0 - mode
	star1 - direction
	star3 - direction
	star5 - direction
	groundstation2 - direction
	star0 - direction
	groundstation4 - direction
	phenomenon6 - direction
	star7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation4)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 infrared0)
	(calibration_target instrument1 star0)
	(calibration_target instrument1 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star7)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation4)
	(supports instrument3 infrared0)
	(calibration_target instrument3 groundstation4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
)
(:goal (and
	(pointing satellite0 phenomenon6)
	(have_image phenomenon6 infrared0)
	(have_image star7 infrared0)
	(have_image planet8 infrared0)
))

)
