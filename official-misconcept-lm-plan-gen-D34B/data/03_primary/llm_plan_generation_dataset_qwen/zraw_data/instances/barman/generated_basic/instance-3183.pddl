(define (problem prob)
 (:domain barman)
 (:objects 
      shaker276 - shaker
      left right - hand
      shot69 - shot
      ingredient402 ingredient105 ingredient199 - ingredient
      cocktail147 - cocktail
      dispenser135 dispenser217 dispenser354 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker276)
  (ontable shot69)
  (dispenses dispenser135 ingredient402)
  (dispenses dispenser217 ingredient105)
  (dispenses dispenser354 ingredient199)
  (clean shaker276)
  (clean shot69)
  (empty shaker276)
  (empty shot69)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker276 l0)
  (shaker-level shaker276 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail147 ingredient105)
  (cocktail-part2 cocktail147 ingredient199)
)
 (:goal
  (and
      (contains shot69 cocktail147)
)))
