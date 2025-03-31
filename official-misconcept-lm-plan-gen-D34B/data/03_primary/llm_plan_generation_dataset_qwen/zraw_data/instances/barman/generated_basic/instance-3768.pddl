(define (problem prob)
 (:domain barman)
 (:objects 
      shaker468 - shaker
      left right - hand
      shot104 shot306 shot9 - shot
      ingredient84 ingredient248 - ingredient
      cocktail8 - cocktail
      dispenser52 dispenser268 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker468)
  (ontable shot104)
  (ontable shot306)
  (ontable shot9)
  (dispenses dispenser52 ingredient84)
  (dispenses dispenser268 ingredient248)
  (clean shaker468)
  (clean shot104)
  (clean shot306)
  (clean shot9)
  (empty shaker468)
  (empty shot104)
  (empty shot306)
  (empty shot9)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker468 l0)
  (shaker-level shaker468 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail8 ingredient84)
  (cocktail-part2 cocktail8 ingredient248)
)
 (:goal
  (and
      (contains shot104 cocktail8)
      (contains shot306 cocktail8)
)))
