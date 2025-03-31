(define (problem prob)
 (:domain barman)
 (:objects 
      shaker468 - shaker
      left right - hand
      shot470 shot147 - shot
      ingredient124 ingredient114 ingredient22 ingredient85 - ingredient
      cocktail223 - cocktail
      dispenser126 dispenser433 dispenser304 dispenser208 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker468)
  (ontable shot470)
  (ontable shot147)
  (dispenses dispenser126 ingredient124)
  (dispenses dispenser433 ingredient114)
  (dispenses dispenser304 ingredient22)
  (dispenses dispenser208 ingredient85)
  (clean shaker468)
  (clean shot470)
  (clean shot147)
  (empty shaker468)
  (empty shot470)
  (empty shot147)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker468 l0)
  (shaker-level shaker468 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail223 ingredient22)
  (cocktail-part2 cocktail223 ingredient124)
)
 (:goal
  (and
      (contains shot470 cocktail223)
)))
