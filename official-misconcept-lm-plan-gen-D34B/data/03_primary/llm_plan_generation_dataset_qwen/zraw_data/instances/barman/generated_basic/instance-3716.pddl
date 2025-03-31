(define (problem prob)
 (:domain barman)
 (:objects 
      shaker440 - shaker
      left right - hand
      shot90 - shot
      ingredient174 ingredient493 ingredient330 ingredient307 - ingredient
      cocktail32 - cocktail
      dispenser245 dispenser157 dispenser496 dispenser149 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker440)
  (ontable shot90)
  (dispenses dispenser245 ingredient174)
  (dispenses dispenser157 ingredient493)
  (dispenses dispenser496 ingredient330)
  (dispenses dispenser149 ingredient307)
  (clean shaker440)
  (clean shot90)
  (empty shaker440)
  (empty shot90)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker440 l0)
  (shaker-level shaker440 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail32 ingredient330)
  (cocktail-part2 cocktail32 ingredient307)
)
 (:goal
  (and
      (contains shot90 cocktail32)
)))
