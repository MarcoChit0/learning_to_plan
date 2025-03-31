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
	instrument6 - instrument
	image1 - mode
	infrared0 - mode
	thermograph2 - mode
	groundstation0 - direction
	star5 - direction
	star1 - direction
	groundstation4 - direction
	groundstation3 - direction
	groundstation2 - direction
	planet6 - direction
	phenomenon7 - direction
	star8 - direction
	star9 - direction
)
(:init
	(supports instrument0 infrared0)
	(calibration_target instrument0 groundstation2)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 image1)
	(supports instrument1 infrared0)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 groundstation3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star9)
	(supports instrument2 infrared0)
	(supports instrument2 image1)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star5)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 groundstation4)
	(calibration_target instrument3 groundstation2)
	(supports instrument4 image1)
	(supports instrument4 infrared0)
	(supports instrument4 thermograph2)
	(calibration_target instrument4 groundstation4)
	(calibration_target instrument4 star1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument5 infrared0)
	(supports instrument5 image1)
	(calibration_target instrument5 groundstation3)
	(supports instrument6 image1)
	(supports instrument6 thermograph2)
	(supports instrument6 infrared0)
	(calibration_target instrument6 groundstation2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
)
(:goal (and
	(pointing satellite0 star8)
	(pointing satellite1 star8)
	(have_image planet6 infrared0)
	(have_image phenomenon7 image1)
	(have_image star8 image1)
	(have_image star9 thermograph2)
))

)
