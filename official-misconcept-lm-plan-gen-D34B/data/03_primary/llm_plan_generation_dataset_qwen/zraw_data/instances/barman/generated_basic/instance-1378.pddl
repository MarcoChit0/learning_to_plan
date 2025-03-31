(define (problem prob)
 (:domain barman)
 (:objects 
      shaker487 - shaker
      left right - hand
      shot406 shot350 shot412 - shot
      ingredient108 ingredient490 ingredient200 ingredient16 - ingredient
      cocktail39 - cocktail
      dispenser253 dispenser306 dispenser412 dispenser413 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker487)
  (ontable shot406)
  (ontable shot350)
  (ontable shot412)
  (dispenses dispenser253 ingredient108)
  (dispenses dispenser306 ingredient490)
  (dispenses dispenser412 ingredient200)
  (dispenses dispenser413 ingredient16)
  (clean shaker487)
  (clean shot406)
  (clean shot350)
  (clean shot412)
  (empty shaker487)
  (empty shot406)
  (empty shot350)
  (empty shot412)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker487 l0)
  (shaker-level shaker487 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail39 ingredient200)
  (cocktail-part2 cocktail39 ingredient108)
)
 (:goal
  (and
      (contains shot406 cocktail39)
      (contains shot350 ingredient490)
)))
