(define (problem prob)
 (:domain barman)
 (:objects 
      shaker165 - shaker
      left right - hand
      shot0 - shot
      ingredient44 ingredient81 ingredient33 - ingredient
      cocktail242 - cocktail
      dispenser119 dispenser134 dispenser396 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker165)
  (ontable shot0)
  (dispenses dispenser119 ingredient44)
  (dispenses dispenser134 ingredient81)
  (dispenses dispenser396 ingredient33)
  (clean shaker165)
  (clean shot0)
  (empty shaker165)
  (empty shot0)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker165 l0)
  (shaker-level shaker165 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail242 ingredient33)
  (cocktail-part2 cocktail242 ingredient44)
)
 (:goal
  (and
      (contains shot0 cocktail242)
)))
