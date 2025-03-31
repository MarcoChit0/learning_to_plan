(define (problem prob)
 (:domain barman)
 (:objects 
      shaker361 - shaker
      left right - hand
      shot466 shot180 - shot
      ingredient228 ingredient456 - ingredient
      cocktail1 - cocktail
      dispenser40 dispenser216 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker361)
  (ontable shot466)
  (ontable shot180)
  (dispenses dispenser40 ingredient228)
  (dispenses dispenser216 ingredient456)
  (clean shaker361)
  (clean shot466)
  (clean shot180)
  (empty shaker361)
  (empty shot466)
  (empty shot180)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker361 l0)
  (shaker-level shaker361 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient228)
  (cocktail-part2 cocktail1 ingredient456)
)
 (:goal
  (and
      (contains shot466 cocktail1)
)))
