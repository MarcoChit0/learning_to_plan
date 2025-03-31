(define (problem prob)
 (:domain barman)
 (:objects 
      shaker390 - shaker
      left right - hand
      shot399 shot317 - shot
      ingredient102 ingredient397 ingredient365 ingredient272 - ingredient
      cocktail1 - cocktail
      dispenser316 dispenser201 dispenser319 dispenser268 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker390)
  (ontable shot399)
  (ontable shot317)
  (dispenses dispenser316 ingredient102)
  (dispenses dispenser201 ingredient397)
  (dispenses dispenser319 ingredient365)
  (dispenses dispenser268 ingredient272)
  (clean shaker390)
  (clean shot399)
  (clean shot317)
  (empty shaker390)
  (empty shot399)
  (empty shot317)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker390 l0)
  (shaker-level shaker390 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient397)
  (cocktail-part2 cocktail1 ingredient102)
)
 (:goal
  (and
      (contains shot399 cocktail1)
)))
