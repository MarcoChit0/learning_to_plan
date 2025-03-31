(define (problem prob)
 (:domain barman)
 (:objects 
      shaker229 - shaker
      left right - hand
      shot92 - shot
      ingredient208 ingredient41 ingredient154 ingredient130 - ingredient
      cocktail322 - cocktail
      dispenser240 dispenser329 dispenser120 dispenser154 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker229)
  (ontable shot92)
  (dispenses dispenser240 ingredient208)
  (dispenses dispenser329 ingredient41)
  (dispenses dispenser120 ingredient154)
  (dispenses dispenser154 ingredient130)
  (clean shaker229)
  (clean shot92)
  (empty shaker229)
  (empty shot92)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker229 l0)
  (shaker-level shaker229 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail322 ingredient41)
  (cocktail-part2 cocktail322 ingredient130)
)
 (:goal
  (and
      (contains shot92 cocktail322)
)))
