(define (problem prob)
 (:domain barman)
 (:objects 
      shaker125 - shaker
      left right - hand
      shot3 shot406 - shot
      ingredient146 ingredient298 ingredient1 ingredient105 - ingredient
      cocktail1 - cocktail
      dispenser362 dispenser20 dispenser109 dispenser397 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker125)
  (ontable shot3)
  (ontable shot406)
  (dispenses dispenser362 ingredient146)
  (dispenses dispenser20 ingredient298)
  (dispenses dispenser109 ingredient1)
  (dispenses dispenser397 ingredient105)
  (clean shaker125)
  (clean shot3)
  (clean shot406)
  (empty shaker125)
  (empty shot3)
  (empty shot406)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker125 l0)
  (shaker-level shaker125 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient298)
  (cocktail-part2 cocktail1 ingredient146)
)
 (:goal
  (and
      (contains shot3 cocktail1)
)))
