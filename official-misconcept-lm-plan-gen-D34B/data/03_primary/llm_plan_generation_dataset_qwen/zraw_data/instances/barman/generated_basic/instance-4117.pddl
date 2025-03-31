(define (problem prob)
 (:domain barman)
 (:objects 
      shaker236 - shaker
      left right - hand
      shot291 shot225 - shot
      ingredient111 ingredient229 ingredient180 - ingredient
      cocktail15 - cocktail
      dispenser428 dispenser281 dispenser364 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker236)
  (ontable shot291)
  (ontable shot225)
  (dispenses dispenser428 ingredient111)
  (dispenses dispenser281 ingredient229)
  (dispenses dispenser364 ingredient180)
  (clean shaker236)
  (clean shot291)
  (clean shot225)
  (empty shaker236)
  (empty shot291)
  (empty shot225)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker236 l0)
  (shaker-level shaker236 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail15 ingredient180)
  (cocktail-part2 cocktail15 ingredient229)
)
 (:goal
  (and
      (contains shot291 cocktail15)
)))
