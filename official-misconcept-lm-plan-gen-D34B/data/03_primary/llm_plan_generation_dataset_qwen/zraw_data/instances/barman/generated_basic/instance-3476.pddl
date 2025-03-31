(define (problem prob)
 (:domain barman)
 (:objects 
      shaker296 - shaker
      left right - hand
      shot455 shot109 shot405 - shot
      ingredient328 ingredient41 ingredient52 - ingredient
      cocktail439 - cocktail
      dispenser291 dispenser63 dispenser409 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker296)
  (ontable shot455)
  (ontable shot109)
  (ontable shot405)
  (dispenses dispenser291 ingredient328)
  (dispenses dispenser63 ingredient41)
  (dispenses dispenser409 ingredient52)
  (clean shaker296)
  (clean shot455)
  (clean shot109)
  (clean shot405)
  (empty shaker296)
  (empty shot455)
  (empty shot109)
  (empty shot405)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker296 l0)
  (shaker-level shaker296 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail439 ingredient52)
  (cocktail-part2 cocktail439 ingredient328)
)
 (:goal
  (and
      (contains shot455 cocktail439)
      (contains shot109 ingredient52)
)))
