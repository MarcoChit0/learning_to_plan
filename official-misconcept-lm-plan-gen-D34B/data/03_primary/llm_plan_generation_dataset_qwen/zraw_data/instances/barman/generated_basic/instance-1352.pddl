(define (problem prob)
 (:domain barman)
 (:objects 
      shaker272 - shaker
      left right - hand
      shot442 shot344 shot94 - shot
      ingredient147 ingredient484 ingredient355 - ingredient
      cocktail117 - cocktail
      dispenser296 dispenser366 dispenser271 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker272)
  (ontable shot442)
  (ontable shot344)
  (ontable shot94)
  (dispenses dispenser296 ingredient147)
  (dispenses dispenser366 ingredient484)
  (dispenses dispenser271 ingredient355)
  (clean shaker272)
  (clean shot442)
  (clean shot344)
  (clean shot94)
  (empty shaker272)
  (empty shot442)
  (empty shot344)
  (empty shot94)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker272 l0)
  (shaker-level shaker272 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail117 ingredient147)
  (cocktail-part2 cocktail117 ingredient355)
)
 (:goal
  (and
      (contains shot442 cocktail117)
      (contains shot344 cocktail117)
)))
