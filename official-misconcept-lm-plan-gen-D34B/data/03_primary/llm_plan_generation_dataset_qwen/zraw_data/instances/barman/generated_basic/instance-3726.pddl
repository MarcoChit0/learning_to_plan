(define (problem prob)
 (:domain barman)
 (:objects 
      shaker343 - shaker
      left right - hand
      shot348 - shot
      ingredient368 ingredient195 ingredient425 ingredient48 - ingredient
      cocktail298 - cocktail
      dispenser408 dispenser421 dispenser347 dispenser445 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker343)
  (ontable shot348)
  (dispenses dispenser408 ingredient368)
  (dispenses dispenser421 ingredient195)
  (dispenses dispenser347 ingredient425)
  (dispenses dispenser445 ingredient48)
  (clean shaker343)
  (clean shot348)
  (empty shaker343)
  (empty shot348)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker343 l0)
  (shaker-level shaker343 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail298 ingredient425)
  (cocktail-part2 cocktail298 ingredient195)
)
 (:goal
  (and
      (contains shot348 cocktail298)
)))
