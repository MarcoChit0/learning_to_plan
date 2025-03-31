(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	infrared1 - mode
	infrared2 - mode
	image0 - mode
	groundstation4 - direction
	star1 - direction
	groundstation5 - direction
	groundstation3 - direction
	star2 - direction
	groundstation0 - direction
	planet6 - direction
	planet7 - direction
	phenomenon8 - direction
	star9 - direction
	star10 - direction
	phenomenon11 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 infrared1)
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation4)
	(calibration_target instrument0 groundstation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation3)
	(supports instrument1 image0)
	(calibration_target instrument1 star1)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 infrared1)
	(supports instrument2 image0)
	(supports instrument2 infrared2)
	(calibration_target instrument2 groundstation5)
	(calibration_target instrument2 star1)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation3)
	(supports instrument3 infrared2)
	(supports instrument3 image0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star2)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 infrared1)
	(supports instrument4 infrared2)
	(calibration_target instrument4 groundstation0)
	(calibration_target instrument4 star2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet7)
)
(:goal (and
	(pointing satellite0 star9)
	(pointing satellite1 groundstation3)
	(have_image planet6 infrared1)
	(have_image planet7 infrared1)
	(have_image phenomenon8 infrared2)
	(have_image star9 infrared2)
	(have_image star10 image0)
	(have_image phenomenon11 infrared2)
))

)
