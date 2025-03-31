(define (problem prob)
 (:domain barman)
 (:objects 
      shaker198 - shaker
      left right - hand
      shot278 - shot
      ingredient119 ingredient386 ingredient225 ingredient214 - ingredient
      cocktail215 - cocktail
      dispenser291 dispenser153 dispenser132 dispenser449 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker198)
  (ontable shot278)
  (dispenses dispenser291 ingredient119)
  (dispenses dispenser153 ingredient386)
  (dispenses dispenser132 ingredient225)
  (dispenses dispenser449 ingredient214)
  (clean shaker198)
  (clean shot278)
  (empty shaker198)
  (empty shot278)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker198 l0)
  (shaker-level shaker198 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail215 ingredient214)
  (cocktail-part2 cocktail215 ingredient225)
)
 (:goal
  (and
      (contains shot278 cocktail215)
)))
