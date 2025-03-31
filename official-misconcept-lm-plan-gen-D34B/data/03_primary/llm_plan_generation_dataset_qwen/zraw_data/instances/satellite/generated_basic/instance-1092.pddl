(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	infrared2 - mode
	infrared1 - mode
	image0 - mode
	groundstation2 - direction
	groundstation4 - direction
	groundstation5 - direction
	star0 - direction
	star3 - direction
	groundstation1 - direction
	planet6 - direction
	star7 - direction
	star8 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 image0)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation5)
	(supports instrument2 infrared1)
	(calibration_target instrument2 groundstation5)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared2)
	(supports instrument3 image0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star7)
	(supports instrument4 infrared2)
	(supports instrument4 image0)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star0)
	(supports instrument5 infrared1)
	(supports instrument5 infrared2)
	(calibration_target instrument5 star3)
	(supports instrument6 image0)
	(supports instrument6 infrared1)
	(calibration_target instrument6 groundstation1)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star3)
)
(:goal (and
	(pointing satellite2 groundstation4)
	(have_image planet6 infrared1)
	(have_image star7 infrared2)
	(have_image star8 image0)
))

)
