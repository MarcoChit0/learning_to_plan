(define (problem prob)
 (:domain barman)
 (:objects 
      shaker109 - shaker
      left right - hand
      shot337 - shot
      ingredient378 ingredient141 - ingredient
      cocktail387 - cocktail
      dispenser485 dispenser394 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker109)
  (ontable shot337)
  (dispenses dispenser485 ingredient378)
  (dispenses dispenser394 ingredient141)
  (clean shaker109)
  (clean shot337)
  (empty shaker109)
  (empty shot337)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker109 l0)
  (shaker-level shaker109 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail387 ingredient141)
  (cocktail-part2 cocktail387 ingredient378)
)
 (:goal
  (and
      (contains shot337 cocktail387)
)))
