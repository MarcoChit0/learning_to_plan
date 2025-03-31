(define (problem prob)
 (:domain barman)
 (:objects 
      shaker484 - shaker
      left right - hand
      shot12 shot29 - shot
      ingredient474 ingredient498 ingredient486 ingredient301 - ingredient
      cocktail177 - cocktail
      dispenser326 dispenser133 dispenser329 dispenser233 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker484)
  (ontable shot12)
  (ontable shot29)
  (dispenses dispenser326 ingredient474)
  (dispenses dispenser133 ingredient498)
  (dispenses dispenser329 ingredient486)
  (dispenses dispenser233 ingredient301)
  (clean shaker484)
  (clean shot12)
  (clean shot29)
  (empty shaker484)
  (empty shot12)
  (empty shot29)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker484 l0)
  (shaker-level shaker484 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail177 ingredient301)
  (cocktail-part2 cocktail177 ingredient498)
)
 (:goal
  (and
      (contains shot12 cocktail177)
)))
