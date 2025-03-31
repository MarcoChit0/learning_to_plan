(define (problem prob)
 (:domain barman)
 (:objects 
      shaker352 - shaker
      left right - hand
      shot221 shot215 shot48 shot242 - shot
      ingredient383 ingredient436 - ingredient
      cocktail390 - cocktail
      dispenser327 dispenser7 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker352)
  (ontable shot221)
  (ontable shot215)
  (ontable shot48)
  (ontable shot242)
  (dispenses dispenser327 ingredient383)
  (dispenses dispenser7 ingredient436)
  (clean shaker352)
  (clean shot221)
  (clean shot215)
  (clean shot48)
  (clean shot242)
  (empty shaker352)
  (empty shot221)
  (empty shot215)
  (empty shot48)
  (empty shot242)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker352 l0)
  (shaker-level shaker352 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail390 ingredient383)
  (cocktail-part2 cocktail390 ingredient436)
)
 (:goal
  (and
      (contains shot221 cocktail390)
      (contains shot215 ingredient436)
      (contains shot48 cocktail390)
)))
