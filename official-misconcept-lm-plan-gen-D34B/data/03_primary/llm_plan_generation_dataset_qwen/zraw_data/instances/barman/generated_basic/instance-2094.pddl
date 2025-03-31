(define (problem prob)
 (:domain barman)
 (:objects 
      shaker29 - shaker
      left right - hand
      shot430 shot301 - shot
      ingredient269 ingredient201 - ingredient
      cocktail172 - cocktail
      dispenser226 dispenser219 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker29)
  (ontable shot430)
  (ontable shot301)
  (dispenses dispenser226 ingredient269)
  (dispenses dispenser219 ingredient201)
  (clean shaker29)
  (clean shot430)
  (clean shot301)
  (empty shaker29)
  (empty shot430)
  (empty shot301)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker29 l0)
  (shaker-level shaker29 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail172 ingredient201)
  (cocktail-part2 cocktail172 ingredient269)
)
 (:goal
  (and
      (contains shot430 cocktail172)
)))
