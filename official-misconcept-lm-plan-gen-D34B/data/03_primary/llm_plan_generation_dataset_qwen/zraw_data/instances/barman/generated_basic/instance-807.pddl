(define (problem prob)
 (:domain barman)
 (:objects 
      shaker435 - shaker
      left right - hand
      shot188 shot64 - shot
      ingredient103 ingredient383 ingredient284 ingredient387 - ingredient
      cocktail1 - cocktail
      dispenser52 dispenser232 dispenser32 dispenser258 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker435)
  (ontable shot188)
  (ontable shot64)
  (dispenses dispenser52 ingredient103)
  (dispenses dispenser232 ingredient383)
  (dispenses dispenser32 ingredient284)
  (dispenses dispenser258 ingredient387)
  (clean shaker435)
  (clean shot188)
  (clean shot64)
  (empty shaker435)
  (empty shot188)
  (empty shot64)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker435 l0)
  (shaker-level shaker435 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient103)
  (cocktail-part2 cocktail1 ingredient383)
)
 (:goal
  (and
      (contains shot188 cocktail1)
)))
