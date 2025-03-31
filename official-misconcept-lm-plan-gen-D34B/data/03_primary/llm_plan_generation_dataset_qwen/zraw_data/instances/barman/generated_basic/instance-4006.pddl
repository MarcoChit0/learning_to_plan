(define (problem prob)
 (:domain barman)
 (:objects 
      shaker435 - shaker
      left right - hand
      shot373 shot302 shot274 - shot
      ingredient493 ingredient111 - ingredient
      cocktail403 - cocktail
      dispenser328 dispenser262 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker435)
  (ontable shot373)
  (ontable shot302)
  (ontable shot274)
  (dispenses dispenser328 ingredient493)
  (dispenses dispenser262 ingredient111)
  (clean shaker435)
  (clean shot373)
  (clean shot302)
  (clean shot274)
  (empty shaker435)
  (empty shot373)
  (empty shot302)
  (empty shot274)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker435 l0)
  (shaker-level shaker435 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail403 ingredient493)
  (cocktail-part2 cocktail403 ingredient111)
)
 (:goal
  (and
      (contains shot373 cocktail403)
      (contains shot302 cocktail403)
)))
