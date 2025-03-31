(define (problem prob)
 (:domain barman)
 (:objects 
      shaker78 - shaker
      left right - hand
      shot247 - shot
      ingredient83 ingredient337 ingredient6 - ingredient
      cocktail183 - cocktail
      dispenser416 dispenser477 dispenser38 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker78)
  (ontable shot247)
  (dispenses dispenser416 ingredient83)
  (dispenses dispenser477 ingredient337)
  (dispenses dispenser38 ingredient6)
  (clean shaker78)
  (clean shot247)
  (empty shaker78)
  (empty shot247)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker78 l0)
  (shaker-level shaker78 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail183 ingredient6)
  (cocktail-part2 cocktail183 ingredient337)
)
 (:goal
  (and
      (contains shot247 cocktail183)
)))
