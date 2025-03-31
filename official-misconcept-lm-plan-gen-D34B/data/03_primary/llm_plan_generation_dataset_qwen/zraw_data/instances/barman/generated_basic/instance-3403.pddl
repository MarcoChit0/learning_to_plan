(define (problem prob)
 (:domain barman)
 (:objects 
      shaker114 - shaker
      left right - hand
      shot341 shot436 - shot
      ingredient433 ingredient491 ingredient13 ingredient58 - ingredient
      cocktail115 - cocktail
      dispenser431 dispenser40 dispenser196 dispenser150 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker114)
  (ontable shot341)
  (ontable shot436)
  (dispenses dispenser431 ingredient433)
  (dispenses dispenser40 ingredient491)
  (dispenses dispenser196 ingredient13)
  (dispenses dispenser150 ingredient58)
  (clean shaker114)
  (clean shot341)
  (clean shot436)
  (empty shaker114)
  (empty shot341)
  (empty shot436)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker114 l0)
  (shaker-level shaker114 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail115 ingredient13)
  (cocktail-part2 cocktail115 ingredient491)
)
 (:goal
  (and
      (contains shot341 cocktail115)
)))
