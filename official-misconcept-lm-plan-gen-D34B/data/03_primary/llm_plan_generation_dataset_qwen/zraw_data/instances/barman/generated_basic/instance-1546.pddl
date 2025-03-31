(define (problem prob)
 (:domain barman)
 (:objects 
      shaker176 - shaker
      left right - hand
      shot355 - shot
      ingredient214 ingredient271 - ingredient
      cocktail106 - cocktail
      dispenser104 dispenser250 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker176)
  (ontable shot355)
  (dispenses dispenser104 ingredient214)
  (dispenses dispenser250 ingredient271)
  (clean shaker176)
  (clean shot355)
  (empty shaker176)
  (empty shot355)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker176 l0)
  (shaker-level shaker176 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail106 ingredient271)
  (cocktail-part2 cocktail106 ingredient214)
)
 (:goal
  (and
      (contains shot355 cocktail106)
)))
