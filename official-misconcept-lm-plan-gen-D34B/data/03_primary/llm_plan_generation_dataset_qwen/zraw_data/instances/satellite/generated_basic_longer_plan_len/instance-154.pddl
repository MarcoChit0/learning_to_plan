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
	image2 - mode
	spectrograph0 - mode
	infrared1 - mode
	groundstation3 - direction
	star2 - direction
	star4 - direction
	groundstation1 - direction
	groundstation0 - direction
	phenomenon5 - direction
	planet6 - direction
	phenomenon7 - direction
	planet8 - direction
	star9 - direction
	planet10 - direction
	phenomenon11 - direction
	star12 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 image2)
	(supports instrument0 infrared1)
	(calibration_target instrument0 star2)
	(supports instrument1 spectrograph0)
	(supports instrument1 image2)
	(supports instrument1 infrared3)
	(calibration_target instrument1 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon7)
	(supports instrument2 image2)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star4)
	(supports instrument3 image2)
	(calibration_target instrument3 groundstation1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 groundstation0)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star2)
)
(:goal (and
	(have_image phenomenon5 image2)
	(have_image planet6 infrared1)
	(have_image phenomenon7 infrared1)
	(have_image planet8 image2)
	(have_image star9 infrared3)
	(have_image planet10 infrared3)
	(have_image phenomenon11 infrared1)
	(have_image star12 infrared1)
))

)
