(define (problem prob)
 (:domain barman)
 (:objects 
      shaker499 - shaker
      left right - hand
      shot492 shot333 shot7 - shot
      ingredient354 ingredient455 ingredient366 ingredient107 - ingredient
      cocktail98 - cocktail
      dispenser307 dispenser162 dispenser484 dispenser412 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker499)
  (ontable shot492)
  (ontable shot333)
  (ontable shot7)
  (dispenses dispenser307 ingredient354)
  (dispenses dispenser162 ingredient455)
  (dispenses dispenser484 ingredient366)
  (dispenses dispenser412 ingredient107)
  (clean shaker499)
  (clean shot492)
  (clean shot333)
  (clean shot7)
  (empty shaker499)
  (empty shot492)
  (empty shot333)
  (empty shot7)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker499 l0)
  (shaker-level shaker499 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail98 ingredient366)
  (cocktail-part2 cocktail98 ingredient354)
)
 (:goal
  (and
      (contains shot492 cocktail98)
      (contains shot333 cocktail98)
)))
