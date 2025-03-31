(define (problem prob)
 (:domain barman)
 (:objects 
      shaker312 - shaker
      left right - hand
      shot110 - shot
      ingredient371 ingredient5 - ingredient
      cocktail463 - cocktail
      dispenser285 dispenser157 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker312)
  (ontable shot110)
  (dispenses dispenser285 ingredient371)
  (dispenses dispenser157 ingredient5)
  (clean shaker312)
  (clean shot110)
  (empty shaker312)
  (empty shot110)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker312 l0)
  (shaker-level shaker312 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail463 ingredient371)
  (cocktail-part2 cocktail463 ingredient5)
)
 (:goal
  (and
      (contains shot110 cocktail463)
)))
