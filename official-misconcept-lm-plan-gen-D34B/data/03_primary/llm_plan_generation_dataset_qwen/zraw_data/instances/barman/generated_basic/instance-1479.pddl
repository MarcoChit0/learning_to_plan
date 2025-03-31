(define (problem prob)
 (:domain barman)
 (:objects 
      shaker210 - shaker
      left right - hand
      shot61 shot378 - shot
      ingredient103 ingredient230 ingredient27 ingredient354 - ingredient
      cocktail372 - cocktail
      dispenser334 dispenser61 dispenser63 dispenser352 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker210)
  (ontable shot61)
  (ontable shot378)
  (dispenses dispenser334 ingredient103)
  (dispenses dispenser61 ingredient230)
  (dispenses dispenser63 ingredient27)
  (dispenses dispenser352 ingredient354)
  (clean shaker210)
  (clean shot61)
  (clean shot378)
  (empty shaker210)
  (empty shot61)
  (empty shot378)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker210 l0)
  (shaker-level shaker210 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail372 ingredient103)
  (cocktail-part2 cocktail372 ingredient354)
)
 (:goal
  (and
      (contains shot61 cocktail372)
)))
