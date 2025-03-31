(define (problem prob)
 (:domain barman)
 (:objects 
      shaker270 - shaker
      left right - hand
      shot486 - shot
      ingredient32 ingredient44 - ingredient
      cocktail313 - cocktail
      dispenser271 dispenser68 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker270)
  (ontable shot486)
  (dispenses dispenser271 ingredient32)
  (dispenses dispenser68 ingredient44)
  (clean shaker270)
  (clean shot486)
  (empty shaker270)
  (empty shot486)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker270 l0)
  (shaker-level shaker270 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail313 ingredient44)
  (cocktail-part2 cocktail313 ingredient32)
)
 (:goal
  (and
      (contains shot486 cocktail313)
)))
