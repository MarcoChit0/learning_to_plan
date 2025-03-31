(define (problem prob)
 (:domain barman)
 (:objects 
      shaker44 - shaker
      left right - hand
      shot384 shot442 - shot
      ingredient137 ingredient146 ingredient469 ingredient273 - ingredient
      cocktail183 - cocktail
      dispenser349 dispenser389 dispenser77 dispenser151 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker44)
  (ontable shot384)
  (ontable shot442)
  (dispenses dispenser349 ingredient137)
  (dispenses dispenser389 ingredient146)
  (dispenses dispenser77 ingredient469)
  (dispenses dispenser151 ingredient273)
  (clean shaker44)
  (clean shot384)
  (clean shot442)
  (empty shaker44)
  (empty shot384)
  (empty shot442)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker44 l0)
  (shaker-level shaker44 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail183 ingredient146)
  (cocktail-part2 cocktail183 ingredient469)
)
 (:goal
  (and
      (contains shot384 cocktail183)
)))
