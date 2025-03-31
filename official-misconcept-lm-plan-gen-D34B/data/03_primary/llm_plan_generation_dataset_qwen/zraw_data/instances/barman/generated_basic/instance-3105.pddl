(define (problem prob)
 (:domain barman)
 (:objects 
      shaker15 - shaker
      left right - hand
      shot466 shot258 - shot
      ingredient171 ingredient378 - ingredient
      cocktail106 - cocktail
      dispenser196 dispenser61 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker15)
  (ontable shot466)
  (ontable shot258)
  (dispenses dispenser196 ingredient171)
  (dispenses dispenser61 ingredient378)
  (clean shaker15)
  (clean shot466)
  (clean shot258)
  (empty shaker15)
  (empty shot466)
  (empty shot258)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker15 l0)
  (shaker-level shaker15 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail106 ingredient378)
  (cocktail-part2 cocktail106 ingredient171)
)
 (:goal
  (and
      (contains shot466 cocktail106)
)))
