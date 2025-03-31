(define (problem prob)
 (:domain barman)
 (:objects 
      shaker368 - shaker
      left right - hand
      shot279 shot44 - shot
      ingredient436 ingredient80 ingredient165 ingredient378 - ingredient
      cocktail1 - cocktail
      dispenser302 dispenser356 dispenser226 dispenser494 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker368)
  (ontable shot279)
  (ontable shot44)
  (dispenses dispenser302 ingredient436)
  (dispenses dispenser356 ingredient80)
  (dispenses dispenser226 ingredient165)
  (dispenses dispenser494 ingredient378)
  (clean shaker368)
  (clean shot279)
  (clean shot44)
  (empty shaker368)
  (empty shot279)
  (empty shot44)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker368 l0)
  (shaker-level shaker368 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient436)
  (cocktail-part2 cocktail1 ingredient378)
)
 (:goal
  (and
      (contains shot279 cocktail1)
)))
