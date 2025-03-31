(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	satellite2 - satellite
	instrument5 - instrument
	thermograph1 - mode
	infrared0 - mode
	image2 - mode
	infrared3 - mode
	groundstation1 - direction
	star3 - direction
	star2 - direction
	groundstation0 - direction
	groundstation5 - direction
	groundstation4 - direction
	planet6 - direction
)
(:init
	(supports instrument0 infrared3)
	(calibration_target instrument0 groundstation4)
	(supports instrument1 image2)
	(supports instrument1 infrared0)
	(supports instrument1 thermograph1)
	(calibration_target instrument1 star2)
	(calibration_target instrument1 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet6)
	(supports instrument2 infrared0)
	(supports instrument2 image2)
	(calibration_target instrument2 star2)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 image2)
	(supports instrument3 infrared3)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 infrared3)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation0)
	(calibration_target instrument4 groundstation4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument5 infrared0)
	(calibration_target instrument5 groundstation4)
	(calibration_target instrument5 groundstation5)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(pointing satellite0 star3)
	(pointing satellite1 groundstation0)
	(pointing satellite2 groundstation0)
	(have_image planet6 image2)
))

)
