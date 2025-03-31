(define (problem prob)
 (:domain barman)
 (:objects 
      shaker279 - shaker
      left right - hand
      shot301 - shot
      ingredient284 ingredient323 ingredient248 ingredient498 - ingredient
      cocktail1 - cocktail
      dispenser303 dispenser25 dispenser84 dispenser310 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker279)
  (ontable shot301)
  (dispenses dispenser303 ingredient284)
  (dispenses dispenser25 ingredient323)
  (dispenses dispenser84 ingredient248)
  (dispenses dispenser310 ingredient498)
  (clean shaker279)
  (clean shot301)
  (empty shaker279)
  (empty shot301)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker279 l0)
  (shaker-level shaker279 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient248)
  (cocktail-part2 cocktail1 ingredient323)
)
 (:goal
  (and
      (contains shot301 cocktail1)
)))
