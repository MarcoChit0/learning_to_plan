(define (problem prob)
 (:domain barman)
 (:objects 
      shaker113 - shaker
      left right - hand
      shot296 shot246 shot104 - shot
      ingredient449 ingredient416 ingredient126 ingredient349 - ingredient
      cocktail468 - cocktail
      dispenser54 dispenser102 dispenser227 dispenser282 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker113)
  (ontable shot296)
  (ontable shot246)
  (ontable shot104)
  (dispenses dispenser54 ingredient449)
  (dispenses dispenser102 ingredient416)
  (dispenses dispenser227 ingredient126)
  (dispenses dispenser282 ingredient349)
  (clean shaker113)
  (clean shot296)
  (clean shot246)
  (clean shot104)
  (empty shaker113)
  (empty shot296)
  (empty shot246)
  (empty shot104)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker113 l0)
  (shaker-level shaker113 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail468 ingredient449)
  (cocktail-part2 cocktail468 ingredient349)
)
 (:goal
  (and
      (contains shot296 cocktail468)
      (contains shot246 cocktail468)
)))
