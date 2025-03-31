(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	satellite3 - satellite
	instrument6 - instrument
	image0 - mode
	star0 - direction
	groundstation5 - direction
	star1 - direction
	groundstation2 - direction
	groundstation3 - direction
	star4 - direction
	star6 - direction
	star7 - direction
	star8 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation5)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 image0)
	(calibration_target instrument2 star0)
	(calibration_target instrument2 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument3 image0)
	(calibration_target instrument3 groundstation5)
	(calibration_target instrument3 star1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star8)
	(supports instrument4 image0)
	(calibration_target instrument4 star1)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation3)
	(calibration_target instrument5 groundstation2)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument6 image0)
	(calibration_target instrument6 star4)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star7)
)
(:goal (and
	(pointing satellite0 star1)
	(have_image star6 image0)
	(have_image star7 image0)
	(have_image star8 image0)
))

)
