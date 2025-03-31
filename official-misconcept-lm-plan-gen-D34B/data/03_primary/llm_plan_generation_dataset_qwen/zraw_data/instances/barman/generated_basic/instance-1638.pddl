(define (problem prob)
 (:domain barman)
 (:objects 
      shaker475 - shaker
      left right - hand
      shot395 shot384 shot79 - shot
      ingredient117 ingredient177 - ingredient
      cocktail196 - cocktail
      dispenser296 dispenser150 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker475)
  (ontable shot395)
  (ontable shot384)
  (ontable shot79)
  (dispenses dispenser296 ingredient117)
  (dispenses dispenser150 ingredient177)
  (clean shaker475)
  (clean shot395)
  (clean shot384)
  (clean shot79)
  (empty shaker475)
  (empty shot395)
  (empty shot384)
  (empty shot79)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker475 l0)
  (shaker-level shaker475 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail196 ingredient177)
  (cocktail-part2 cocktail196 ingredient117)
)
 (:goal
  (and
      (contains shot395 cocktail196)
      (contains shot384 ingredient177)
)))
