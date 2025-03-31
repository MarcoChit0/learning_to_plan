(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite3 - satellite
	instrument4 - instrument
	spectrograph0 - mode
	star1 - direction
	groundstation2 - direction
	groundstation0 - direction
	star4 - direction
	groundstation3 - direction
	star5 - direction
	phenomenon6 - direction
	planet7 - direction
	planet8 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 groundstation0)
	(calibration_target instrument0 star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon6)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 star4)
	(calibration_target instrument1 groundstation3)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 groundstation3)
	(calibration_target instrument2 star5)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star4)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star5)
	(calibration_target instrument4 groundstation3)
	(on_board instrument4 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star5)
)
(:goal (and
	(pointing satellite1 star1)
	(have_image phenomenon6 spectrograph0)
	(have_image planet7 spectrograph0)
	(have_image planet8 spectrograph0)
))

)
