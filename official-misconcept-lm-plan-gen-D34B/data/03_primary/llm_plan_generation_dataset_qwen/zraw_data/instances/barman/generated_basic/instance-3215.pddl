(define (problem prob)
 (:domain barman)
 (:objects 
      shaker11 - shaker
      left right - hand
      shot337 - shot
      ingredient459 ingredient32 - ingredient
      cocktail30 - cocktail
      dispenser392 dispenser206 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker11)
  (ontable shot337)
  (dispenses dispenser392 ingredient459)
  (dispenses dispenser206 ingredient32)
  (clean shaker11)
  (clean shot337)
  (empty shaker11)
  (empty shot337)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker11 l0)
  (shaker-level shaker11 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail30 ingredient459)
  (cocktail-part2 cocktail30 ingredient32)
)
 (:goal
  (and
      (contains shot337 cocktail30)
)))
