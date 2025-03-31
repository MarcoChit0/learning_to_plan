(define (problem prob)
 (:domain barman)
 (:objects 
      shaker184 - shaker
      left right - hand
      shot66 shot409 - shot
      ingredient24 ingredient91 ingredient364 ingredient419 - ingredient
      cocktail299 - cocktail
      dispenser371 dispenser74 dispenser33 dispenser167 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker184)
  (ontable shot66)
  (ontable shot409)
  (dispenses dispenser371 ingredient24)
  (dispenses dispenser74 ingredient91)
  (dispenses dispenser33 ingredient364)
  (dispenses dispenser167 ingredient419)
  (clean shaker184)
  (clean shot66)
  (clean shot409)
  (empty shaker184)
  (empty shot66)
  (empty shot409)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker184 l0)
  (shaker-level shaker184 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail299 ingredient91)
  (cocktail-part2 cocktail299 ingredient419)
)
 (:goal
  (and
      (contains shot66 cocktail299)
)))
