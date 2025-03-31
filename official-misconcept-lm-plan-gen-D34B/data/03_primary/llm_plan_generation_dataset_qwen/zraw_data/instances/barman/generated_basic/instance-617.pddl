(define (problem prob)
 (:domain barman)
 (:objects 
      shaker478 - shaker
      left right - hand
      shot38 shot458 - shot
      ingredient230 ingredient103 ingredient270 ingredient175 - ingredient
      cocktail1 - cocktail
      dispenser125 dispenser269 dispenser299 dispenser374 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker478)
  (ontable shot38)
  (ontable shot458)
  (dispenses dispenser125 ingredient230)
  (dispenses dispenser269 ingredient103)
  (dispenses dispenser299 ingredient270)
  (dispenses dispenser374 ingredient175)
  (clean shaker478)
  (clean shot38)
  (clean shot458)
  (empty shaker478)
  (empty shot38)
  (empty shot458)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker478 l0)
  (shaker-level shaker478 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient230)
  (cocktail-part2 cocktail1 ingredient270)
)
 (:goal
  (and
      (contains shot38 cocktail1)
)))
