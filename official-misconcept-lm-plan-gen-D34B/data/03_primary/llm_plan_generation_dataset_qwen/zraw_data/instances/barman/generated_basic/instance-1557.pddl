(define (problem prob)
 (:domain barman)
 (:objects 
      shaker209 - shaker
      left right - hand
      shot162 - shot
      ingredient439 ingredient391 - ingredient
      cocktail400 - cocktail
      dispenser481 dispenser303 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker209)
  (ontable shot162)
  (dispenses dispenser481 ingredient439)
  (dispenses dispenser303 ingredient391)
  (clean shaker209)
  (clean shot162)
  (empty shaker209)
  (empty shot162)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker209 l0)
  (shaker-level shaker209 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail400 ingredient439)
  (cocktail-part2 cocktail400 ingredient391)
)
 (:goal
  (and
      (contains shot162 cocktail400)
)))
