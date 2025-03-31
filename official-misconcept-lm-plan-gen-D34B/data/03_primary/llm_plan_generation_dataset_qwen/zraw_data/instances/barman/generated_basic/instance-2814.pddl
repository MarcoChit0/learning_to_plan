(define (problem prob)
 (:domain barman)
 (:objects 
      shaker488 - shaker
      left right - hand
      shot250 shot413 - shot
      ingredient423 ingredient384 - ingredient
      cocktail245 - cocktail
      dispenser452 dispenser91 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker488)
  (ontable shot250)
  (ontable shot413)
  (dispenses dispenser452 ingredient423)
  (dispenses dispenser91 ingredient384)
  (clean shaker488)
  (clean shot250)
  (clean shot413)
  (empty shaker488)
  (empty shot250)
  (empty shot413)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker488 l0)
  (shaker-level shaker488 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail245 ingredient423)
  (cocktail-part2 cocktail245 ingredient384)
)
 (:goal
  (and
      (contains shot250 cocktail245)
)))
