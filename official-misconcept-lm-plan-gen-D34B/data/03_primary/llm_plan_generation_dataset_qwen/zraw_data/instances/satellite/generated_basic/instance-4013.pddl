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
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	instrument6 - instrument
	image2 - mode
	spectrograph1 - mode
	thermograph0 - mode
	groundstation0 - direction
	star1 - direction
	groundstation2 - direction
	star3 - direction
)
(:init
	(supports instrument0 spectrograph1)
	(supports instrument0 image2)
	(calibration_target instrument0 groundstation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star1)
	(supports instrument1 image2)
	(supports instrument1 spectrograph1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 groundstation0)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 groundstation0)
	(supports instrument2 spectrograph1)
	(supports instrument2 image2)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 groundstation0)
	(supports instrument3 thermograph0)
	(supports instrument3 spectrograph1)
	(calibration_target instrument3 groundstation0)
	(supports instrument4 thermograph0)
	(supports instrument4 spectrograph1)
	(supports instrument4 image2)
	(calibration_target instrument4 star1)
	(on_board instrument2 satellite2)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation2)
	(supports instrument5 spectrograph1)
	(calibration_target instrument5 star1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 groundstation2)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(power_avail satellite3)
	(pointing satellite3 star1)
)
(:goal (and
	(pointing satellite2 star1)
	(pointing satellite3 star3)
	(have_image star3 spectrograph1)
))

)
