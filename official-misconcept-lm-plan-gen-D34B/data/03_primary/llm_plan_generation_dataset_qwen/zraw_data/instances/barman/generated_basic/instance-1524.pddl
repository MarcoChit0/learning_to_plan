(define (problem prob)
 (:domain barman)
 (:objects 
      shaker126 - shaker
      left right - hand
      shot358 shot120 - shot
      ingredient252 ingredient38 ingredient392 ingredient288 - ingredient
      cocktail439 - cocktail
      dispenser291 dispenser352 dispenser143 dispenser400 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker126)
  (ontable shot358)
  (ontable shot120)
  (dispenses dispenser291 ingredient252)
  (dispenses dispenser352 ingredient38)
  (dispenses dispenser143 ingredient392)
  (dispenses dispenser400 ingredient288)
  (clean shaker126)
  (clean shot358)
  (clean shot120)
  (empty shaker126)
  (empty shot358)
  (empty shot120)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker126 l0)
  (shaker-level shaker126 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail439 ingredient288)
  (cocktail-part2 cocktail439 ingredient252)
)
 (:goal
  (and
      (contains shot358 cocktail439)
)))
