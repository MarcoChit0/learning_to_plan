(define (problem prob)
 (:domain barman)
 (:objects 
      shaker100 - shaker
      left right - hand
      shot475 shot209 - shot
      ingredient386 ingredient61 ingredient180 ingredient104 - ingredient
      cocktail1 - cocktail
      dispenser313 dispenser149 dispenser302 dispenser154 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker100)
  (ontable shot475)
  (ontable shot209)
  (dispenses dispenser313 ingredient386)
  (dispenses dispenser149 ingredient61)
  (dispenses dispenser302 ingredient180)
  (dispenses dispenser154 ingredient104)
  (clean shaker100)
  (clean shot475)
  (clean shot209)
  (empty shaker100)
  (empty shot475)
  (empty shot209)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker100 l0)
  (shaker-level shaker100 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient104)
  (cocktail-part2 cocktail1 ingredient386)
)
 (:goal
  (and
      (contains shot475 cocktail1)
)))
