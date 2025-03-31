(define (problem prob)
 (:domain barman)
 (:objects 
      shaker85 - shaker
      left right - hand
      shot437 shot402 - shot
      ingredient125 ingredient42 ingredient183 - ingredient
      cocktail1 - cocktail
      dispenser400 dispenser83 dispenser169 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker85)
  (ontable shot437)
  (ontable shot402)
  (dispenses dispenser400 ingredient125)
  (dispenses dispenser83 ingredient42)
  (dispenses dispenser169 ingredient183)
  (clean shaker85)
  (clean shot437)
  (clean shot402)
  (empty shaker85)
  (empty shot437)
  (empty shot402)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker85 l0)
  (shaker-level shaker85 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient42)
  (cocktail-part2 cocktail1 ingredient183)
)
 (:goal
  (and
      (contains shot437 cocktail1)
)))
