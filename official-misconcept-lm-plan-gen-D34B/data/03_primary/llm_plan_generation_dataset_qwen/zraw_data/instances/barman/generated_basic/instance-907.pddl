(define (problem prob)
 (:domain barman)
 (:objects 
      shaker241 - shaker
      left right - hand
      shot459 shot80 - shot
      ingredient31 ingredient310 - ingredient
      cocktail1 - cocktail
      dispenser25 dispenser433 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker241)
  (ontable shot459)
  (ontable shot80)
  (dispenses dispenser25 ingredient31)
  (dispenses dispenser433 ingredient310)
  (clean shaker241)
  (clean shot459)
  (clean shot80)
  (empty shaker241)
  (empty shot459)
  (empty shot80)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker241 l0)
  (shaker-level shaker241 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient310)
  (cocktail-part2 cocktail1 ingredient31)
)
 (:goal
  (and
      (contains shot459 cocktail1)
)))
