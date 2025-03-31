(define (problem prob)
 (:domain barman)
 (:objects 
      shaker18 - shaker
      left right - hand
      shot138 shot423 shot126 - shot
      ingredient166 ingredient488 ingredient273 ingredient194 - ingredient
      cocktail243 - cocktail
      dispenser71 dispenser161 dispenser465 dispenser33 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker18)
  (ontable shot138)
  (ontable shot423)
  (ontable shot126)
  (dispenses dispenser71 ingredient166)
  (dispenses dispenser161 ingredient488)
  (dispenses dispenser465 ingredient273)
  (dispenses dispenser33 ingredient194)
  (clean shaker18)
  (clean shot138)
  (clean shot423)
  (clean shot126)
  (empty shaker18)
  (empty shot138)
  (empty shot423)
  (empty shot126)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker18 l0)
  (shaker-level shaker18 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail243 ingredient488)
  (cocktail-part2 cocktail243 ingredient194)
)
 (:goal
  (and
      (contains shot138 cocktail243)
      (contains shot423 cocktail243)
)))
