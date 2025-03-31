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
	image1 - mode
	image2 - mode
	thermograph0 - mode
	groundstation0 - direction
	star1 - direction
	groundstation5 - direction
	groundstation4 - direction
	star2 - direction
	star3 - direction
	phenomenon6 - direction
	star7 - direction
	planet8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image1)
	(calibration_target instrument0 groundstation5)
	(supports instrument1 image2)
	(supports instrument1 image1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation5)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation0)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument3 image1)
	(calibration_target instrument3 star2)
	(calibration_target instrument3 groundstation5)
	(supports instrument4 image1)
	(calibration_target instrument4 groundstation5)
	(supports instrument5 image2)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation5)
	(calibration_target instrument5 star1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument6 thermograph0)
	(supports instrument6 image1)
	(calibration_target instrument6 star2)
	(calibration_target instrument6 groundstation4)
	(supports instrument7 image1)
	(supports instrument7 thermograph0)
	(supports instrument7 image2)
	(calibration_target instrument7 star2)
	(supports instrument8 thermograph0)
	(supports instrument8 image2)
	(supports instrument8 image1)
	(calibration_target instrument8 star3)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation5)
)
(:goal (and
	(pointing satellite1 star1)
	(pointing satellite2 groundstation4)
	(have_image phenomenon6 thermograph0)
	(have_image star7 image2)
	(have_image planet8 image2)
	(have_image planet9 image1)
))

)
