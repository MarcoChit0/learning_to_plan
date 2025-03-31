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
	image2 - mode
	infrared3 - mode
	infrared1 - mode
	spectrograph0 - mode
	star2 - direction
	groundstation1 - direction
	star0 - direction
	phenomenon3 - direction
	planet4 - direction
	phenomenon5 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 image2)
	(calibration_target instrument0 star2)
	(supports instrument1 spectrograph0)
	(supports instrument1 image2)
	(supports instrument1 infrared3)
	(calibration_target instrument1 star0)
	(supports instrument2 infrared1)
	(supports instrument2 infrared3)
	(calibration_target instrument2 star2)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon5)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 image2)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation1)
	(supports instrument5 image2)
	(supports instrument5 spectrograph0)
	(supports instrument5 infrared1)
	(calibration_target instrument5 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation1)
)
(:goal (and
	(have_image phenomenon3 spectrograph0)
	(have_image planet4 infrared3)
	(have_image phenomenon5 infrared1)
))

)
