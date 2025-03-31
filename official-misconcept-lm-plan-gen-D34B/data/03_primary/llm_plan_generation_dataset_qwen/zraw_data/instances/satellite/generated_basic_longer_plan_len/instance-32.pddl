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
	satellite2 - satellite
	instrument5 - instrument
	instrument6 - instrument
	satellite3 - satellite
	instrument7 - instrument
	infrared1 - mode
	infrared3 - mode
	thermograph2 - mode
	image0 - mode
	star3 - direction
	star0 - direction
	star1 - direction
	groundstation2 - direction
	phenomenon4 - direction
	phenomenon5 - direction
	phenomenon6 - direction
	planet7 - direction
	star8 - direction
	phenomenon9 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star3)
	(supports instrument1 infrared3)
	(supports instrument1 image0)
	(supports instrument1 infrared1)
	(calibration_target instrument1 star1)
	(supports instrument2 image0)
	(calibration_target instrument2 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument3 thermograph2)
	(calibration_target instrument3 star3)
	(supports instrument4 thermograph2)
	(supports instrument4 infrared1)
	(supports instrument4 infrared3)
	(calibration_target instrument4 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation2)
	(supports instrument5 infrared3)
	(supports instrument5 image0)
	(calibration_target instrument5 groundstation2)
	(supports instrument6 image0)
	(supports instrument6 infrared3)
	(supports instrument6 thermograph2)
	(calibration_target instrument6 star1)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument7 infrared3)
	(supports instrument7 infrared1)
	(supports instrument7 image0)
	(calibration_target instrument7 groundstation2)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 groundstation2)
)
(:goal (and
	(pointing satellite0 phenomenon4)
	(pointing satellite2 star3)
	(have_image phenomenon4 image0)
	(have_image phenomenon5 thermograph2)
	(have_image phenomenon6 infrared1)
	(have_image planet7 infrared3)
	(have_image star8 image0)
	(have_image phenomenon9 infrared3)
))

)
