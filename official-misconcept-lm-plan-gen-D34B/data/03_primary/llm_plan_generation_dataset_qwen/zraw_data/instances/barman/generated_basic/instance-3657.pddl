(define (problem prob)
 (:domain barman)
 (:objects 
      shaker21 - shaker
      left right - hand
      shot60 shot153 - shot
      ingredient25 ingredient425 ingredient303 ingredient372 - ingredient
      cocktail195 - cocktail
      dispenser150 dispenser12 dispenser290 dispenser212 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker21)
  (ontable shot60)
  (ontable shot153)
  (dispenses dispenser150 ingredient25)
  (dispenses dispenser12 ingredient425)
  (dispenses dispenser290 ingredient303)
  (dispenses dispenser212 ingredient372)
  (clean shaker21)
  (clean shot60)
  (clean shot153)
  (empty shaker21)
  (empty shot60)
  (empty shot153)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker21 l0)
  (shaker-level shaker21 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail195 ingredient25)
  (cocktail-part2 cocktail195 ingredient372)
)
 (:goal
  (and
      (contains shot60 cocktail195)
)))
