(define (problem prob)
 (:domain barman)
 (:objects 
      shaker0 - shaker
      left right - hand
      shot225 shot79 shot64 - shot
      ingredient187 ingredient270 ingredient37 ingredient496 - ingredient
      cocktail151 - cocktail
      dispenser438 dispenser230 dispenser447 dispenser432 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker0)
  (ontable shot225)
  (ontable shot79)
  (ontable shot64)
  (dispenses dispenser438 ingredient187)
  (dispenses dispenser230 ingredient270)
  (dispenses dispenser447 ingredient37)
  (dispenses dispenser432 ingredient496)
  (clean shaker0)
  (clean shot225)
  (clean shot79)
  (clean shot64)
  (empty shaker0)
  (empty shot225)
  (empty shot79)
  (empty shot64)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker0 l0)
  (shaker-level shaker0 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail151 ingredient270)
  (cocktail-part2 cocktail151 ingredient187)
)
 (:goal
  (and
      (contains shot225 cocktail151)
      (contains shot79 cocktail151)
)))
