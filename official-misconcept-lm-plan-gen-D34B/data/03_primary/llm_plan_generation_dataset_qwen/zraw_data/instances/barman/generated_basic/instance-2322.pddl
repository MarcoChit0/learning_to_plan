(define (problem prob)
 (:domain barman)
 (:objects 
      shaker301 - shaker
      left right - hand
      shot331 - shot
      ingredient335 ingredient217 - ingredient
      cocktail407 - cocktail
      dispenser17 dispenser3 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker301)
  (ontable shot331)
  (dispenses dispenser17 ingredient335)
  (dispenses dispenser3 ingredient217)
  (clean shaker301)
  (clean shot331)
  (empty shaker301)
  (empty shot331)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker301 l0)
  (shaker-level shaker301 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail407 ingredient335)
  (cocktail-part2 cocktail407 ingredient217)
)
 (:goal
  (and
      (contains shot331 cocktail407)
)))
