(define (problem prob)
 (:domain barman)
 (:objects 
      shaker116 - shaker
      left right - hand
      shot242 shot204 - shot
      ingredient173 ingredient107 ingredient33 - ingredient
      cocktail1 - cocktail
      dispenser382 dispenser124 dispenser405 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker116)
  (ontable shot242)
  (ontable shot204)
  (dispenses dispenser382 ingredient173)
  (dispenses dispenser124 ingredient107)
  (dispenses dispenser405 ingredient33)
  (clean shaker116)
  (clean shot242)
  (clean shot204)
  (empty shaker116)
  (empty shot242)
  (empty shot204)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker116 l0)
  (shaker-level shaker116 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient107)
  (cocktail-part2 cocktail1 ingredient173)
)
 (:goal
  (and
      (contains shot242 cocktail1)
)))
