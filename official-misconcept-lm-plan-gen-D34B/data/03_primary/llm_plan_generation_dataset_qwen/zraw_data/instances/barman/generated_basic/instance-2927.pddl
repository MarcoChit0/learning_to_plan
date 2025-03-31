(define (problem prob)
 (:domain barman)
 (:objects 
      shaker286 - shaker
      left right - hand
      shot90 - shot
      ingredient371 ingredient355 - ingredient
      cocktail158 - cocktail
      dispenser306 dispenser128 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker286)
  (ontable shot90)
  (dispenses dispenser306 ingredient371)
  (dispenses dispenser128 ingredient355)
  (clean shaker286)
  (clean shot90)
  (empty shaker286)
  (empty shot90)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker286 l0)
  (shaker-level shaker286 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail158 ingredient355)
  (cocktail-part2 cocktail158 ingredient371)
)
 (:goal
  (and
      (contains shot90 cocktail158)
)))
