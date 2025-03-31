(define (problem prob)
 (:domain barman)
 (:objects 
      shaker39 - shaker
      left right - hand
      shot267 - shot
      ingredient165 ingredient128 - ingredient
      cocktail191 - cocktail
      dispenser455 dispenser100 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker39)
  (ontable shot267)
  (dispenses dispenser455 ingredient165)
  (dispenses dispenser100 ingredient128)
  (clean shaker39)
  (clean shot267)
  (empty shaker39)
  (empty shot267)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker39 l0)
  (shaker-level shaker39 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail191 ingredient165)
  (cocktail-part2 cocktail191 ingredient128)
)
 (:goal
  (and
      (contains shot267 cocktail191)
)))
