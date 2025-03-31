(define (problem prob)
 (:domain barman)
 (:objects 
      shaker496 - shaker
      left right - hand
      shot378 shot156 - shot
      ingredient112 ingredient215 - ingredient
      cocktail1 - cocktail
      dispenser478 dispenser157 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker496)
  (ontable shot378)
  (ontable shot156)
  (dispenses dispenser478 ingredient112)
  (dispenses dispenser157 ingredient215)
  (clean shaker496)
  (clean shot378)
  (clean shot156)
  (empty shaker496)
  (empty shot378)
  (empty shot156)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker496 l0)
  (shaker-level shaker496 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient112)
  (cocktail-part2 cocktail1 ingredient215)
)
 (:goal
  (and
      (contains shot378 cocktail1)
)))
