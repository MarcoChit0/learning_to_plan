(define (problem prob)
 (:domain barman)
 (:objects 
      shaker329 - shaker
      left right - hand
      shot20 - shot
      ingredient207 ingredient47 ingredient302 ingredient49 - ingredient
      cocktail1 - cocktail
      dispenser117 dispenser41 dispenser51 dispenser355 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker329)
  (ontable shot20)
  (dispenses dispenser117 ingredient207)
  (dispenses dispenser41 ingredient47)
  (dispenses dispenser51 ingredient302)
  (dispenses dispenser355 ingredient49)
  (clean shaker329)
  (clean shot20)
  (empty shaker329)
  (empty shot20)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker329 l0)
  (shaker-level shaker329 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient49)
  (cocktail-part2 cocktail1 ingredient302)
)
 (:goal
  (and
      (contains shot20 cocktail1)
)))
