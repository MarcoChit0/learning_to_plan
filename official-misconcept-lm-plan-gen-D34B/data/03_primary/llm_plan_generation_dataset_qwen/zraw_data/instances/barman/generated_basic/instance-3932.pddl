(define (problem prob)
 (:domain barman)
 (:objects 
      shaker438 - shaker
      left right - hand
      shot456 shot59 - shot
      ingredient401 ingredient467 - ingredient
      cocktail61 - cocktail
      dispenser396 dispenser193 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker438)
  (ontable shot456)
  (ontable shot59)
  (dispenses dispenser396 ingredient401)
  (dispenses dispenser193 ingredient467)
  (clean shaker438)
  (clean shot456)
  (clean shot59)
  (empty shaker438)
  (empty shot456)
  (empty shot59)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker438 l0)
  (shaker-level shaker438 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail61 ingredient467)
  (cocktail-part2 cocktail61 ingredient401)
)
 (:goal
  (and
      (contains shot456 cocktail61)
)))
