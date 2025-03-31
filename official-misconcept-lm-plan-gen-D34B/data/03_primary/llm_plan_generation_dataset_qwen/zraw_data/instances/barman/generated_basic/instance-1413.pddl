(define (problem prob)
 (:domain barman)
 (:objects 
      shaker45 - shaker
      left right - hand
      shot19 - shot
      ingredient470 ingredient224 ingredient108 ingredient343 - ingredient
      cocktail331 - cocktail
      dispenser498 dispenser244 dispenser62 dispenser7 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker45)
  (ontable shot19)
  (dispenses dispenser498 ingredient470)
  (dispenses dispenser244 ingredient224)
  (dispenses dispenser62 ingredient108)
  (dispenses dispenser7 ingredient343)
  (clean shaker45)
  (clean shot19)
  (empty shaker45)
  (empty shot19)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker45 l0)
  (shaker-level shaker45 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail331 ingredient343)
  (cocktail-part2 cocktail331 ingredient470)
)
 (:goal
  (and
      (contains shot19 cocktail331)
)))
