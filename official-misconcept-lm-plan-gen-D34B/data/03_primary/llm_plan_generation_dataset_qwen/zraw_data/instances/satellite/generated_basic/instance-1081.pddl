(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	infrared2 - mode
	infrared0 - mode
	infrared1 - mode
	groundstation1 - direction
	groundstation5 - direction
	groundstation0 - direction
	groundstation3 - direction
	groundstation4 - direction
	groundstation2 - direction
	planet6 - direction
	phenomenon7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 groundstation1)
	(supports instrument1 infrared2)
	(calibration_target instrument1 groundstation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet8)
	(supports instrument2 infrared2)
	(supports instrument2 infrared0)
	(calibration_target instrument2 groundstation0)
	(calibration_target instrument2 groundstation5)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 planet8)
	(supports instrument3 infrared1)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 infrared2)
	(supports instrument4 infrared0)
	(calibration_target instrument4 groundstation4)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation4)
)
(:goal (and
	(pointing satellite0 phenomenon7)
	(pointing satellite1 planet8)
	(pointing satellite2 groundstation1)
	(have_image planet6 infrared0)
	(have_image phenomenon7 infrared0)
	(have_image planet8 infrared0)
))

)
