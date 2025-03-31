(define (problem prob)
 (:domain barman)
 (:objects 
      shaker15 - shaker
      left right - hand
      shot73 shot9 - shot
      ingredient132 ingredient15 - ingredient
      cocktail173 - cocktail
      dispenser62 dispenser57 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker15)
  (ontable shot73)
  (ontable shot9)
  (dispenses dispenser62 ingredient132)
  (dispenses dispenser57 ingredient15)
  (clean shaker15)
  (clean shot73)
  (clean shot9)
  (empty shaker15)
  (empty shot73)
  (empty shot9)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker15 l0)
  (shaker-level shaker15 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail173 ingredient15)
  (cocktail-part2 cocktail173 ingredient132)
)
 (:goal
  (and
      (contains shot73 cocktail173)
)))
