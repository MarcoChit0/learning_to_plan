(define (problem prob)
 (:domain barman)
 (:objects 
      shaker260 - shaker
      left right - hand
      shot204 - shot
      ingredient145 ingredient477 - ingredient
      cocktail367 - cocktail
      dispenser254 dispenser61 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker260)
  (ontable shot204)
  (dispenses dispenser254 ingredient145)
  (dispenses dispenser61 ingredient477)
  (clean shaker260)
  (clean shot204)
  (empty shaker260)
  (empty shot204)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker260 l0)
  (shaker-level shaker260 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail367 ingredient145)
  (cocktail-part2 cocktail367 ingredient477)
)
 (:goal
  (and
      (contains shot204 cocktail367)
)))
