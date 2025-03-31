(define (problem prob)
 (:domain barman)
 (:objects 
      shaker209 - shaker
      left right - hand
      shot97 shot373 shot102 - shot
      ingredient30 ingredient427 - ingredient
      cocktail142 - cocktail
      dispenser439 dispenser74 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker209)
  (ontable shot97)
  (ontable shot373)
  (ontable shot102)
  (dispenses dispenser439 ingredient30)
  (dispenses dispenser74 ingredient427)
  (clean shaker209)
  (clean shot97)
  (clean shot373)
  (clean shot102)
  (empty shaker209)
  (empty shot97)
  (empty shot373)
  (empty shot102)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker209 l0)
  (shaker-level shaker209 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail142 ingredient30)
  (cocktail-part2 cocktail142 ingredient427)
)
 (:goal
  (and
      (contains shot97 cocktail142)
      (contains shot373 cocktail142)
)))
