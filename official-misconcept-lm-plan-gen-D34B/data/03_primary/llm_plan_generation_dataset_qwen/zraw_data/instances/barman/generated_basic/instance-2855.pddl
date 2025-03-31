(define (problem prob)
 (:domain barman)
 (:objects 
      shaker399 - shaker
      left right - hand
      shot301 shot347 - shot
      ingredient433 ingredient353 ingredient230 ingredient337 - ingredient
      cocktail430 - cocktail
      dispenser353 dispenser405 dispenser10 dispenser218 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker399)
  (ontable shot301)
  (ontable shot347)
  (dispenses dispenser353 ingredient433)
  (dispenses dispenser405 ingredient353)
  (dispenses dispenser10 ingredient230)
  (dispenses dispenser218 ingredient337)
  (clean shaker399)
  (clean shot301)
  (clean shot347)
  (empty shaker399)
  (empty shot301)
  (empty shot347)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker399 l0)
  (shaker-level shaker399 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail430 ingredient353)
  (cocktail-part2 cocktail430 ingredient433)
)
 (:goal
  (and
      (contains shot301 cocktail430)
)))
