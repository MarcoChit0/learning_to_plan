(define (problem prob)
 (:domain barman)
 (:objects 
      shaker197 - shaker
      left right - hand
      shot463 shot301 shot386 - shot
      ingredient395 ingredient54 ingredient403 - ingredient
      cocktail319 - cocktail
      dispenser278 dispenser41 dispenser108 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker197)
  (ontable shot463)
  (ontable shot301)
  (ontable shot386)
  (dispenses dispenser278 ingredient395)
  (dispenses dispenser41 ingredient54)
  (dispenses dispenser108 ingredient403)
  (clean shaker197)
  (clean shot463)
  (clean shot301)
  (clean shot386)
  (empty shaker197)
  (empty shot463)
  (empty shot301)
  (empty shot386)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker197 l0)
  (shaker-level shaker197 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail319 ingredient54)
  (cocktail-part2 cocktail319 ingredient403)
)
 (:goal
  (and
      (contains shot463 cocktail319)
      (contains shot301 cocktail319)
)))
