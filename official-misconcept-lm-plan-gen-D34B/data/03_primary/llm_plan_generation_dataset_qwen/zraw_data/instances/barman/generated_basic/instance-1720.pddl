(define (problem prob)
 (:domain barman)
 (:objects 
      shaker274 - shaker
      left right - hand
      shot147 shot142 - shot
      ingredient478 ingredient318 ingredient129 ingredient237 - ingredient
      cocktail57 - cocktail
      dispenser62 dispenser74 dispenser66 dispenser359 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker274)
  (ontable shot147)
  (ontable shot142)
  (dispenses dispenser62 ingredient478)
  (dispenses dispenser74 ingredient318)
  (dispenses dispenser66 ingredient129)
  (dispenses dispenser359 ingredient237)
  (clean shaker274)
  (clean shot147)
  (clean shot142)
  (empty shaker274)
  (empty shot147)
  (empty shot142)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker274 l0)
  (shaker-level shaker274 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail57 ingredient237)
  (cocktail-part2 cocktail57 ingredient318)
)
 (:goal
  (and
      (contains shot147 cocktail57)
)))
