(define (problem prob)
 (:domain barman)
 (:objects 
      shaker301 - shaker
      left right - hand
      shot71 shot409 - shot
      ingredient281 ingredient241 ingredient151 - ingredient
      cocktail1 - cocktail
      dispenser368 dispenser455 dispenser56 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker301)
  (ontable shot71)
  (ontable shot409)
  (dispenses dispenser368 ingredient281)
  (dispenses dispenser455 ingredient241)
  (dispenses dispenser56 ingredient151)
  (clean shaker301)
  (clean shot71)
  (clean shot409)
  (empty shaker301)
  (empty shot71)
  (empty shot409)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker301 l0)
  (shaker-level shaker301 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient241)
  (cocktail-part2 cocktail1 ingredient151)
)
 (:goal
  (and
      (contains shot71 cocktail1)
)))
