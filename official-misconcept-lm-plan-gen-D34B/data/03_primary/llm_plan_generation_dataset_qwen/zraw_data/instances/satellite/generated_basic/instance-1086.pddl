(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	instrument8 - instrument
	infrared1 - mode
	image2 - mode
	image0 - mode
	groundstation0 - direction
	groundstation3 - direction
	star2 - direction
	groundstation5 - direction
	star1 - direction
	star4 - direction
	planet6 - direction
	planet7 - direction
	star8 - direction
)
(:init
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star2)
	(supports instrument2 image0)
	(supports instrument2 image2)
	(calibration_target instrument2 groundstation0)
	(calibration_target instrument2 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet7)
	(supports instrument3 image2)
	(supports instrument3 image0)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star2)
	(supports instrument4 image0)
	(calibration_target instrument4 star2)
	(calibration_target instrument4 groundstation3)
	(supports instrument5 infrared1)
	(calibration_target instrument5 groundstation5)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star8)
	(supports instrument6 image2)
	(calibration_target instrument6 star2)
	(supports instrument7 infrared1)
	(supports instrument7 image2)
	(supports instrument7 image0)
	(calibration_target instrument7 star1)
	(calibration_target instrument7 groundstation5)
	(supports instrument8 infrared1)
	(supports instrument8 image0)
	(calibration_target instrument8 star4)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
)
(:goal (and
	(pointing satellite0 star8)
	(pointing satellite1 groundstation0)
	(have_image planet6 image0)
	(have_image planet7 image0)
	(have_image star8 image2)
))

)
