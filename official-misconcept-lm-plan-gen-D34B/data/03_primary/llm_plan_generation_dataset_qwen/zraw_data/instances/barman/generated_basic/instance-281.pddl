(define (problem prob)
 (:domain barman)
 (:objects 
      shaker341 - shaker
      left right - hand
      shot24 shot474 - shot
      ingredient462 ingredient301 ingredient157 - ingredient
      cocktail1 - cocktail
      dispenser9 dispenser69 dispenser277 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker341)
  (ontable shot24)
  (ontable shot474)
  (dispenses dispenser9 ingredient462)
  (dispenses dispenser69 ingredient301)
  (dispenses dispenser277 ingredient157)
  (clean shaker341)
  (clean shot24)
  (clean shot474)
  (empty shaker341)
  (empty shot24)
  (empty shot474)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker341 l0)
  (shaker-level shaker341 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient157)
  (cocktail-part2 cocktail1 ingredient462)
)
 (:goal
  (and
      (contains shot24 cocktail1)
)))
