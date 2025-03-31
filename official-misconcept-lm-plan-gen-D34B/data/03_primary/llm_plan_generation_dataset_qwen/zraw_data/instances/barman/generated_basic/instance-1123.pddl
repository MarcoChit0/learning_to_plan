(define (problem prob)
 (:domain barman)
 (:objects 
      shaker277 - shaker
      left right - hand
      shot304 shot338 shot301 - shot
      ingredient64 ingredient313 ingredient277 - ingredient
      cocktail1 - cocktail
      dispenser53 dispenser476 dispenser450 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker277)
  (ontable shot304)
  (ontable shot338)
  (ontable shot301)
  (dispenses dispenser53 ingredient64)
  (dispenses dispenser476 ingredient313)
  (dispenses dispenser450 ingredient277)
  (clean shaker277)
  (clean shot304)
  (clean shot338)
  (clean shot301)
  (empty shaker277)
  (empty shot304)
  (empty shot338)
  (empty shot301)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker277 l0)
  (shaker-level shaker277 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient277)
  (cocktail-part2 cocktail1 ingredient64)
)
 (:goal
  (and
      (contains shot304 cocktail1)
      (contains shot338 cocktail1)
)))
