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
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	image1 - mode
	image0 - mode
	star1 - direction
	groundstation2 - direction
	groundstation0 - direction
	star3 - direction
	planet4 - direction
	phenomenon5 - direction
	star6 - direction
	star7 - direction
)
(:init
	(supports instrument0 image0)
	(calibration_target instrument0 groundstation2)
	(supports instrument1 image1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation0)
	(supports instrument2 image0)
	(supports instrument2 image1)
	(calibration_target instrument2 groundstation2)
	(supports instrument3 image0)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 image1)
	(supports instrument4 image0)
	(calibration_target instrument4 star1)
	(supports instrument5 image0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation2)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument6 image0)
	(supports instrument6 image1)
	(calibration_target instrument6 star1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation2)
	(supports instrument8 image1)
	(supports instrument8 image0)
	(calibration_target instrument8 groundstation0)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 planet4)
)
(:goal (and
	(pointing satellite0 planet4)
	(have_image star3 image0)
	(have_image planet4 image1)
	(have_image phenomenon5 image1)
	(have_image star6 image1)
	(have_image star7 image0)
))

)
