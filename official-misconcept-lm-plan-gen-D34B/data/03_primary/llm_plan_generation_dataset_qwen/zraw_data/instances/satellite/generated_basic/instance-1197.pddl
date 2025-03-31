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
	infrared3 - mode
	spectrograph1 - mode
	image0 - mode
	infrared2 - mode
	star1 - direction
	star2 - direction
	star0 - direction
	phenomenon3 - direction
	star4 - direction
	planet5 - direction
)
(:init
	(supports instrument0 infrared3)
	(supports instrument0 spectrograph1)
	(calibration_target instrument0 star2)
	(supports instrument1 image0)
	(supports instrument1 infrared3)
	(supports instrument1 infrared2)
	(calibration_target instrument1 star1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star2)
	(supports instrument2 infrared2)
	(calibration_target instrument2 star0)
	(supports instrument3 infrared2)
	(supports instrument3 image0)
	(supports instrument3 infrared3)
	(calibration_target instrument3 star2)
	(supports instrument4 infrared2)
	(supports instrument4 image0)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 star0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon3)
)
(:goal (and
	(pointing satellite0 phenomenon3)
	(have_image phenomenon3 infrared2)
	(have_image star4 infrared3)
	(have_image planet5 spectrograph1)
))

)
