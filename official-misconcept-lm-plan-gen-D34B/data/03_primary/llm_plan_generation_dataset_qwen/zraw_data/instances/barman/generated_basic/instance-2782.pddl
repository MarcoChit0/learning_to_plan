(define (problem prob)
 (:domain barman)
 (:objects 
      shaker271 - shaker
      left right - hand
      shot49 - shot
      ingredient430 ingredient230 ingredient255 ingredient124 - ingredient
      cocktail220 - cocktail
      dispenser167 dispenser40 dispenser118 dispenser422 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker271)
  (ontable shot49)
  (dispenses dispenser167 ingredient430)
  (dispenses dispenser40 ingredient230)
  (dispenses dispenser118 ingredient255)
  (dispenses dispenser422 ingredient124)
  (clean shaker271)
  (clean shot49)
  (empty shaker271)
  (empty shot49)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker271 l0)
  (shaker-level shaker271 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail220 ingredient430)
  (cocktail-part2 cocktail220 ingredient230)
)
 (:goal
  (and
      (contains shot49 cocktail220)
)))
