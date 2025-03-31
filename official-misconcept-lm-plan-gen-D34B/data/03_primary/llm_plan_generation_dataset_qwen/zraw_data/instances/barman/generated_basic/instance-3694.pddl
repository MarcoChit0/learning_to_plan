(define (problem prob)
 (:domain barman)
 (:objects 
      shaker441 - shaker
      left right - hand
      shot142 shot53 - shot
      ingredient8 ingredient82 ingredient492 - ingredient
      cocktail249 - cocktail
      dispenser73 dispenser330 dispenser391 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker441)
  (ontable shot142)
  (ontable shot53)
  (dispenses dispenser73 ingredient8)
  (dispenses dispenser330 ingredient82)
  (dispenses dispenser391 ingredient492)
  (clean shaker441)
  (clean shot142)
  (clean shot53)
  (empty shaker441)
  (empty shot142)
  (empty shot53)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker441 l0)
  (shaker-level shaker441 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail249 ingredient492)
  (cocktail-part2 cocktail249 ingredient82)
)
 (:goal
  (and
      (contains shot142 cocktail249)
)))
