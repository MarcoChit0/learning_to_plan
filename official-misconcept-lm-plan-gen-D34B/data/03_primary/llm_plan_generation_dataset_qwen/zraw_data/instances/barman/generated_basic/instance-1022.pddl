(define (problem prob)
 (:domain barman)
 (:objects 
      shaker431 - shaker
      left right - hand
      shot61 - shot
      ingredient130 ingredient469 ingredient314 - ingredient
      cocktail1 - cocktail
      dispenser333 dispenser305 dispenser273 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker431)
  (ontable shot61)
  (dispenses dispenser333 ingredient130)
  (dispenses dispenser305 ingredient469)
  (dispenses dispenser273 ingredient314)
  (clean shaker431)
  (clean shot61)
  (empty shaker431)
  (empty shot61)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker431 l0)
  (shaker-level shaker431 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient130)
  (cocktail-part2 cocktail1 ingredient469)
)
 (:goal
  (and
      (contains shot61 cocktail1)
)))
