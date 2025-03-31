(define (problem prob)
 (:domain barman)
 (:objects 
      shaker202 - shaker
      left right - hand
      shot394 shot67 - shot
      ingredient192 ingredient424 ingredient108 - ingredient
      cocktail327 - cocktail
      dispenser425 dispenser24 dispenser42 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker202)
  (ontable shot394)
  (ontable shot67)
  (dispenses dispenser425 ingredient192)
  (dispenses dispenser24 ingredient424)
  (dispenses dispenser42 ingredient108)
  (clean shaker202)
  (clean shot394)
  (clean shot67)
  (empty shaker202)
  (empty shot394)
  (empty shot67)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker202 l0)
  (shaker-level shaker202 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail327 ingredient108)
  (cocktail-part2 cocktail327 ingredient424)
)
 (:goal
  (and
      (contains shot394 cocktail327)
)))
