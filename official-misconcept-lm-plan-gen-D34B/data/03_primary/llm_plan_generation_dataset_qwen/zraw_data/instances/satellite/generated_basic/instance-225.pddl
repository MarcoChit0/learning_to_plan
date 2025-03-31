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
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	satellite4 - satellite
	instrument6 - instrument
	instrument7 - instrument
	image1 - mode
	image0 - mode
	groundstation3 - direction
	groundstation2 - direction
	star1 - direction
	star4 - direction
	groundstation0 - direction
	star5 - direction
	planet6 - direction
	planet7 - direction
	phenomenon8 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation2)
	(supports instrument1 image1)
	(supports instrument1 image0)
	(calibration_target instrument1 groundstation2)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument2 image1)
	(supports instrument2 image0)
	(calibration_target instrument2 star1)
	(supports instrument3 image0)
	(supports instrument3 image1)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 image0)
	(supports instrument4 image1)
	(calibration_target instrument4 star1)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon8)
	(supports instrument5 image1)
	(calibration_target instrument5 star4)
	(on_board instrument5 satellite3)
	(power_avail satellite3)
	(pointing satellite3 planet7)
	(supports instrument6 image1)
	(calibration_target instrument6 star4)
	(supports instrument7 image1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation0)
	(on_board instrument6 satellite4)
	(on_board instrument7 satellite4)
	(power_avail satellite4)
	(pointing satellite4 planet6)
)
(:goal (and
	(pointing satellite2 star5)
	(pointing satellite3 groundstation0)
	(have_image star5 image0)
	(have_image planet6 image1)
	(have_image planet7 image1)
	(have_image phenomenon8 image0)
))

)
