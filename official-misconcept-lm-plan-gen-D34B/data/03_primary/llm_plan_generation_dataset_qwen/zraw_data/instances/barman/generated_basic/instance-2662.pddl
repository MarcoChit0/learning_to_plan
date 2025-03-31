(define (problem prob)
 (:domain barman)
 (:objects 
      shaker406 - shaker
      left right - hand
      shot280 - shot
      ingredient91 ingredient7 ingredient162 ingredient437 - ingredient
      cocktail301 - cocktail
      dispenser323 dispenser417 dispenser498 dispenser21 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker406)
  (ontable shot280)
  (dispenses dispenser323 ingredient91)
  (dispenses dispenser417 ingredient7)
  (dispenses dispenser498 ingredient162)
  (dispenses dispenser21 ingredient437)
  (clean shaker406)
  (clean shot280)
  (empty shaker406)
  (empty shot280)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker406 l0)
  (shaker-level shaker406 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail301 ingredient91)
  (cocktail-part2 cocktail301 ingredient437)
)
 (:goal
  (and
      (contains shot280 cocktail301)
)))
