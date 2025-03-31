(define (problem prob)
 (:domain barman)
 (:objects 
      shaker456 - shaker
      left right - hand
      shot466 shot98 - shot
      ingredient382 ingredient330 ingredient494 ingredient300 - ingredient
      cocktail85 - cocktail
      dispenser189 dispenser229 dispenser149 dispenser35 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker456)
  (ontable shot466)
  (ontable shot98)
  (dispenses dispenser189 ingredient382)
  (dispenses dispenser229 ingredient330)
  (dispenses dispenser149 ingredient494)
  (dispenses dispenser35 ingredient300)
  (clean shaker456)
  (clean shot466)
  (clean shot98)
  (empty shaker456)
  (empty shot466)
  (empty shot98)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker456 l0)
  (shaker-level shaker456 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail85 ingredient494)
  (cocktail-part2 cocktail85 ingredient382)
)
 (:goal
  (and
      (contains shot466 cocktail85)
)))
