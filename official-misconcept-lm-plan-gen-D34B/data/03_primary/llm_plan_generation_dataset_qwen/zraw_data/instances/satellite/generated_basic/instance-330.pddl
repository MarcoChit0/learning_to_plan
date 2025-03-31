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
	instrument6 - instrument
	spectrograph0 - mode
	infrared3 - mode
	spectrograph2 - mode
	thermograph1 - mode
	star2 - direction
	groundstation4 - direction
	star3 - direction
	star1 - direction
	star5 - direction
	groundstation0 - direction
	star6 - direction
)
(:init
	(supports instrument0 infrared3)
	(calibration_target instrument0 groundstation0)
	(calibration_target instrument0 star2)
	(supports instrument1 infrared3)
	(calibration_target instrument1 star1)
	(supports instrument2 thermograph1)
	(calibration_target instrument2 groundstation4)
	(calibration_target instrument2 star5)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 groundstation4)
	(supports instrument3 thermograph1)
	(supports instrument3 spectrograph2)
	(calibration_target instrument3 star5)
	(calibration_target instrument3 groundstation0)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star3)
	(supports instrument4 spectrograph2)
	(supports instrument4 thermograph1)
	(supports instrument4 spectrograph0)
	(calibration_target instrument4 star3)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 star1)
	(calibration_target instrument5 groundstation0)
	(supports instrument6 spectrograph2)
	(supports instrument6 infrared3)
	(supports instrument6 spectrograph0)
	(calibration_target instrument6 groundstation0)
	(calibration_target instrument6 star5)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(on_board instrument6 satellite2)
	(power_avail satellite2)
	(pointing satellite2 groundstation0)
)
(:goal (and
	(pointing satellite0 star3)
	(have_image star6 infrared3)
))

)
