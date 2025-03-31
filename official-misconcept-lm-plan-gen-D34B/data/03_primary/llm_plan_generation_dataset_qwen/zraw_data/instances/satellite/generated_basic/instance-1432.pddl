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
	spectrograph0 - mode
	image2 - mode
	infrared1 - mode
	image3 - mode
	star0 - direction
	phenomenon1 - direction
	phenomenon2 - direction
	planet3 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph0)
	(supports instrument0 image2)
	(calibration_target instrument0 star0)
	(supports instrument1 spectrograph0)
	(supports instrument1 image3)
	(supports instrument1 image2)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star0)
	(supports instrument2 infrared1)
	(supports instrument2 image2)
	(supports instrument2 image3)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared1)
	(supports instrument3 image2)
	(calibration_target instrument3 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon2)
	(supports instrument4 image2)
	(supports instrument4 image3)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star0)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon1)
)
(:goal (and
	(pointing satellite1 phenomenon2)
	(have_image phenomenon1 image3)
	(have_image phenomenon2 infrared1)
	(have_image planet3 infrared1)
))

)
