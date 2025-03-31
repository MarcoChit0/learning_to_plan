(define (problem prob)
 (:domain barman)
 (:objects 
      shaker202 - shaker
      left right - hand
      shot225 shot134 - shot
      ingredient355 ingredient190 - ingredient
      cocktail127 - cocktail
      dispenser242 dispenser417 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker202)
  (ontable shot225)
  (ontable shot134)
  (dispenses dispenser242 ingredient355)
  (dispenses dispenser417 ingredient190)
  (clean shaker202)
  (clean shot225)
  (clean shot134)
  (empty shaker202)
  (empty shot225)
  (empty shot134)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker202 l0)
  (shaker-level shaker202 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail127 ingredient190)
  (cocktail-part2 cocktail127 ingredient355)
)
 (:goal
  (and
      (contains shot225 cocktail127)
)))
