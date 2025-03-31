(define (problem prob)
 (:domain barman)
 (:objects 
      shaker111 - shaker
      left right - hand
      shot162 shot250 - shot
      ingredient172 ingredient402 - ingredient
      cocktail257 - cocktail
      dispenser344 dispenser73 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker111)
  (ontable shot162)
  (ontable shot250)
  (dispenses dispenser344 ingredient172)
  (dispenses dispenser73 ingredient402)
  (clean shaker111)
  (clean shot162)
  (clean shot250)
  (empty shaker111)
  (empty shot162)
  (empty shot250)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker111 l0)
  (shaker-level shaker111 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail257 ingredient402)
  (cocktail-part2 cocktail257 ingredient172)
)
 (:goal
  (and
      (contains shot162 cocktail257)
)))
