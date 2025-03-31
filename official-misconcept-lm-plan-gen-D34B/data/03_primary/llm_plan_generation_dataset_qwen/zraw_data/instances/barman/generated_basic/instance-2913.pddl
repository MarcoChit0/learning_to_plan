(define (problem prob)
 (:domain barman)
 (:objects 
      shaker335 - shaker
      left right - hand
      shot116 - shot
      ingredient337 ingredient301 - ingredient
      cocktail122 - cocktail
      dispenser132 dispenser211 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker335)
  (ontable shot116)
  (dispenses dispenser132 ingredient337)
  (dispenses dispenser211 ingredient301)
  (clean shaker335)
  (clean shot116)
  (empty shaker335)
  (empty shot116)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker335 l0)
  (shaker-level shaker335 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail122 ingredient301)
  (cocktail-part2 cocktail122 ingredient337)
)
 (:goal
  (and
      (contains shot116 cocktail122)
)))
