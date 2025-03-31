(define (problem prob)
 (:domain barman)
 (:objects 
      shaker221 - shaker
      left right - hand
      shot494 shot484 - shot
      ingredient30 ingredient27 ingredient425 ingredient127 - ingredient
      cocktail82 - cocktail
      dispenser37 dispenser469 dispenser85 dispenser64 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker221)
  (ontable shot494)
  (ontable shot484)
  (dispenses dispenser37 ingredient30)
  (dispenses dispenser469 ingredient27)
  (dispenses dispenser85 ingredient425)
  (dispenses dispenser64 ingredient127)
  (clean shaker221)
  (clean shot494)
  (clean shot484)
  (empty shaker221)
  (empty shot494)
  (empty shot484)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker221 l0)
  (shaker-level shaker221 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail82 ingredient127)
  (cocktail-part2 cocktail82 ingredient425)
)
 (:goal
  (and
      (contains shot494 cocktail82)
)))
