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
	spectrograph0 - mode
	infrared1 - mode
	image2 - mode
	spectrograph3 - mode
	star0 - direction
	star2 - direction
	groundstation1 - direction
	groundstation5 - direction
	groundstation3 - direction
	star4 - direction
	phenomenon6 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 spectrograph0)
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 groundstation1)
	(calibration_target instrument0 groundstation3)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 groundstation5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation1)
	(supports instrument2 infrared1)
	(calibration_target instrument2 star4)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation5)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 groundstation3)
	(supports instrument4 spectrograph0)
	(supports instrument4 spectrograph3)
	(supports instrument4 image2)
	(calibration_target instrument4 star4)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation3)
)
(:goal (and
	(pointing satellite2 groundstation5)
	(have_image phenomenon6 image2)
))

)
