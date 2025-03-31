(define (problem prob)
 (:domain barman)
 (:objects 
      shaker114 - shaker
      left right - hand
      shot448 - shot
      ingredient224 ingredient496 ingredient202 ingredient411 - ingredient
      cocktail120 - cocktail
      dispenser475 dispenser185 dispenser151 dispenser225 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker114)
  (ontable shot448)
  (dispenses dispenser475 ingredient224)
  (dispenses dispenser185 ingredient496)
  (dispenses dispenser151 ingredient202)
  (dispenses dispenser225 ingredient411)
  (clean shaker114)
  (clean shot448)
  (empty shaker114)
  (empty shot448)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker114 l0)
  (shaker-level shaker114 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail120 ingredient411)
  (cocktail-part2 cocktail120 ingredient496)
)
 (:goal
  (and
      (contains shot448 cocktail120)
)))
