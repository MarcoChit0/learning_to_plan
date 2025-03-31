(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	image0 - mode
	image3 - mode
	spectrograph2 - mode
	infrared1 - mode
	star0 - direction
	planet1 - direction
	phenomenon2 - direction
	phenomenon3 - direction
)
(:init
	(supports instrument0 image0)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 star0)
	(supports instrument1 image0)
	(calibration_target instrument1 star0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 planet1)
	(supports instrument2 infrared1)
	(supports instrument2 image3)
	(calibration_target instrument2 star0)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star0)
	(supports instrument3 image3)
	(supports instrument3 infrared1)
	(calibration_target instrument3 star0)
	(supports instrument4 image0)
	(supports instrument4 image3)
	(supports instrument4 infrared1)
	(calibration_target instrument4 star0)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 phenomenon2)
)
(:goal (and
	(pointing satellite0 phenomenon2)
	(have_image planet1 image0)
	(have_image phenomenon2 image3)
	(have_image phenomenon3 infrared1)
))

)
