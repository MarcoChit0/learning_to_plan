(define (problem prob)
 (:domain barman)
 (:objects 
      shaker463 - shaker
      left right - hand
      shot301 shot358 - shot
      ingredient225 ingredient157 - ingredient
      cocktail454 - cocktail
      dispenser182 dispenser433 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker463)
  (ontable shot301)
  (ontable shot358)
  (dispenses dispenser182 ingredient225)
  (dispenses dispenser433 ingredient157)
  (clean shaker463)
  (clean shot301)
  (clean shot358)
  (empty shaker463)
  (empty shot301)
  (empty shot358)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker463 l0)
  (shaker-level shaker463 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail454 ingredient225)
  (cocktail-part2 cocktail454 ingredient157)
)
 (:goal
  (and
      (contains shot301 cocktail454)
)))
