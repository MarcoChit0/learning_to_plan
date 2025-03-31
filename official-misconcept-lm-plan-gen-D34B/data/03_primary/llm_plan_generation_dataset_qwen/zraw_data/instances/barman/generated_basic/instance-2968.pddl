(define (problem prob)
 (:domain barman)
 (:objects 
      shaker426 - shaker
      left right - hand
      shot301 - shot
      ingredient250 ingredient173 - ingredient
      cocktail419 - cocktail
      dispenser402 dispenser143 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker426)
  (ontable shot301)
  (dispenses dispenser402 ingredient250)
  (dispenses dispenser143 ingredient173)
  (clean shaker426)
  (clean shot301)
  (empty shaker426)
  (empty shot301)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker426 l0)
  (shaker-level shaker426 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail419 ingredient173)
  (cocktail-part2 cocktail419 ingredient250)
)
 (:goal
  (and
      (contains shot301 cocktail419)
)))
