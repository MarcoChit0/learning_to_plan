(define (problem prob)
 (:domain barman)
 (:objects 
      shaker356 - shaker
      left right - hand
      shot85 - shot
      ingredient10 ingredient230 - ingredient
      cocktail338 - cocktail
      dispenser325 dispenser256 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker356)
  (ontable shot85)
  (dispenses dispenser325 ingredient10)
  (dispenses dispenser256 ingredient230)
  (clean shaker356)
  (clean shot85)
  (empty shaker356)
  (empty shot85)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker356 l0)
  (shaker-level shaker356 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail338 ingredient230)
  (cocktail-part2 cocktail338 ingredient10)
)
 (:goal
  (and
      (contains shot85 cocktail338)
)))
