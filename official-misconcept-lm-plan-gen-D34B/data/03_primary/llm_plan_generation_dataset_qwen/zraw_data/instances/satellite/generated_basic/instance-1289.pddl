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
	infrared0 - mode
	image1 - mode
	spectrograph2 - mode
	star3 - direction
	groundstation2 - direction
	star4 - direction
	star5 - direction
	star0 - direction
	star1 - direction
	phenomenon6 - direction
	planet7 - direction
	star8 - direction
	planet9 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star1)
	(calibration_target instrument0 star4)
	(supports instrument1 infrared0)
	(supports instrument1 spectrograph2)
	(supports instrument1 image1)
	(calibration_target instrument1 star5)
	(supports instrument2 infrared0)
	(supports instrument2 spectrograph2)
	(supports instrument2 image1)
	(calibration_target instrument2 star4)
	(calibration_target instrument2 groundstation2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument3 image1)
	(calibration_target instrument3 star4)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon6)
	(supports instrument4 image1)
	(supports instrument4 infrared0)
	(calibration_target instrument4 star5)
	(supports instrument5 spectrograph2)
	(supports instrument5 infrared0)
	(supports instrument5 image1)
	(calibration_target instrument5 star1)
	(calibration_target instrument5 star0)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star5)
)
(:goal (and
	(pointing satellite0 star8)
	(pointing satellite2 star4)
	(have_image phenomenon6 spectrograph2)
	(have_image planet7 infrared0)
	(have_image star8 infrared0)
	(have_image planet9 infrared0)
))

)
