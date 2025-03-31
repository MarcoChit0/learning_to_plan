(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	infrared1 - mode
	infrared0 - mode
	groundstation0 - direction
	star2 - direction
	groundstation4 - direction
	groundstation3 - direction
	groundstation1 - direction
	star5 - direction
	planet6 - direction
	phenomenon7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 infrared1)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument2 infrared0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star5)
)
(:goal (and
	(have_image star5 infrared1)
	(have_image planet6 infrared0)
	(have_image phenomenon7 infrared1)
	(have_image phenomenon8 infrared0)
))

)
