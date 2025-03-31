(define (problem prob)
 (:domain barman)
 (:objects 
      shaker100 - shaker
      left right - hand
      shot415 shot53 shot224 - shot
      ingredient67 ingredient33 - ingredient
      cocktail53 - cocktail
      dispenser165 dispenser79 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker100)
  (ontable shot415)
  (ontable shot53)
  (ontable shot224)
  (dispenses dispenser165 ingredient67)
  (dispenses dispenser79 ingredient33)
  (clean shaker100)
  (clean shot415)
  (clean shot53)
  (clean shot224)
  (empty shaker100)
  (empty shot415)
  (empty shot53)
  (empty shot224)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker100 l0)
  (shaker-level shaker100 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail53 ingredient67)
  (cocktail-part2 cocktail53 ingredient33)
)
 (:goal
  (and
      (contains shot415 cocktail53)
      (contains shot53 cocktail53)
)))
