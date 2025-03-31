(define (problem prob)
 (:domain barman)
 (:objects 
      shaker305 - shaker
      left right - hand
      shot57 shot380 shot334 - shot
      ingredient390 ingredient483 - ingredient
      cocktail124 - cocktail
      dispenser18 dispenser487 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker305)
  (ontable shot57)
  (ontable shot380)
  (ontable shot334)
  (dispenses dispenser18 ingredient390)
  (dispenses dispenser487 ingredient483)
  (clean shaker305)
  (clean shot57)
  (clean shot380)
  (clean shot334)
  (empty shaker305)
  (empty shot57)
  (empty shot380)
  (empty shot334)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker305 l0)
  (shaker-level shaker305 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail124 ingredient483)
  (cocktail-part2 cocktail124 ingredient390)
)
 (:goal
  (and
      (contains shot57 cocktail124)
      (contains shot380 cocktail124)
)))
