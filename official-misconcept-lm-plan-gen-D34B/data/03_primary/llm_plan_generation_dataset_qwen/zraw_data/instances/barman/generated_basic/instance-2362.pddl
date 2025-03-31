(define (problem prob)
 (:domain barman)
 (:objects 
      shaker488 - shaker
      left right - hand
      shot468 shot106 - shot
      ingredient334 ingredient77 ingredient4 ingredient432 - ingredient
      cocktail56 - cocktail
      dispenser239 dispenser451 dispenser492 dispenser455 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker488)
  (ontable shot468)
  (ontable shot106)
  (dispenses dispenser239 ingredient334)
  (dispenses dispenser451 ingredient77)
  (dispenses dispenser492 ingredient4)
  (dispenses dispenser455 ingredient432)
  (clean shaker488)
  (clean shot468)
  (clean shot106)
  (empty shaker488)
  (empty shot468)
  (empty shot106)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker488 l0)
  (shaker-level shaker488 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail56 ingredient334)
  (cocktail-part2 cocktail56 ingredient432)
)
 (:goal
  (and
      (contains shot468 cocktail56)
)))
