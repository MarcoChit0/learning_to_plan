(define (problem prob)
 (:domain barman)
 (:objects 
      shaker460 - shaker
      left right - hand
      shot385 - shot
      ingredient143 ingredient21 ingredient428 - ingredient
      cocktail413 - cocktail
      dispenser72 dispenser372 dispenser467 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker460)
  (ontable shot385)
  (dispenses dispenser72 ingredient143)
  (dispenses dispenser372 ingredient21)
  (dispenses dispenser467 ingredient428)
  (clean shaker460)
  (clean shot385)
  (empty shaker460)
  (empty shot385)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker460 l0)
  (shaker-level shaker460 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail413 ingredient428)
  (cocktail-part2 cocktail413 ingredient21)
)
 (:goal
  (and
      (contains shot385 cocktail413)
)))
