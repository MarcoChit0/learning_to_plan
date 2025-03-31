(define (problem prob)
 (:domain barman)
 (:objects 
      shaker454 - shaker
      left right - hand
      shot94 shot150 - shot
      ingredient435 ingredient82 - ingredient
      cocktail286 - cocktail
      dispenser329 dispenser426 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker454)
  (ontable shot94)
  (ontable shot150)
  (dispenses dispenser329 ingredient435)
  (dispenses dispenser426 ingredient82)
  (clean shaker454)
  (clean shot94)
  (clean shot150)
  (empty shaker454)
  (empty shot94)
  (empty shot150)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker454 l0)
  (shaker-level shaker454 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail286 ingredient435)
  (cocktail-part2 cocktail286 ingredient82)
)
 (:goal
  (and
      (contains shot94 cocktail286)
)))
