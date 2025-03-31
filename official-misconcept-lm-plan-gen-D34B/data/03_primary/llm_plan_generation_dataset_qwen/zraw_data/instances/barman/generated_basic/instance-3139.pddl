(define (problem prob)
 (:domain barman)
 (:objects 
      shaker49 - shaker
      left right - hand
      shot492 shot404 shot172 - shot
      ingredient399 ingredient428 - ingredient
      cocktail160 - cocktail
      dispenser232 dispenser25 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker49)
  (ontable shot492)
  (ontable shot404)
  (ontable shot172)
  (dispenses dispenser232 ingredient399)
  (dispenses dispenser25 ingredient428)
  (clean shaker49)
  (clean shot492)
  (clean shot404)
  (clean shot172)
  (empty shaker49)
  (empty shot492)
  (empty shot404)
  (empty shot172)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker49 l0)
  (shaker-level shaker49 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail160 ingredient399)
  (cocktail-part2 cocktail160 ingredient428)
)
 (:goal
  (and
      (contains shot492 cocktail160)
      (contains shot404 ingredient428)
)))
