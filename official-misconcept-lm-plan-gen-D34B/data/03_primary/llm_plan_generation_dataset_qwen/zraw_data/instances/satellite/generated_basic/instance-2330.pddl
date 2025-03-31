(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	infrared2 - mode
	thermograph1 - mode
	infrared0 - mode
	star1 - direction
	star3 - direction
	groundstation4 - direction
	groundstation0 - direction
	groundstation2 - direction
	star5 - direction
	planet6 - direction
	phenomenon7 - direction
	planet8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument1 infrared2)
	(supports instrument1 thermograph1)
	(supports instrument1 infrared0)
	(calibration_target instrument1 groundstation0)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
)
(:goal (and
	(have_image star5 thermograph1)
	(have_image planet6 infrared0)
	(have_image phenomenon7 infrared0)
	(have_image planet8 thermograph1)
	(have_image planet9 thermograph1)
))

)
