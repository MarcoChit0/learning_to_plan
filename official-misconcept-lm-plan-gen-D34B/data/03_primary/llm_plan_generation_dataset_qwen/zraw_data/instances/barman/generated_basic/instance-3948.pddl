(define (problem prob)
 (:domain barman)
 (:objects 
      shaker344 - shaker
      left right - hand
      shot404 shot435 - shot
      ingredient296 ingredient143 - ingredient
      cocktail242 - cocktail
      dispenser27 dispenser399 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker344)
  (ontable shot404)
  (ontable shot435)
  (dispenses dispenser27 ingredient296)
  (dispenses dispenser399 ingredient143)
  (clean shaker344)
  (clean shot404)
  (clean shot435)
  (empty shaker344)
  (empty shot404)
  (empty shot435)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker344 l0)
  (shaker-level shaker344 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail242 ingredient143)
  (cocktail-part2 cocktail242 ingredient296)
)
 (:goal
  (and
      (contains shot404 cocktail242)
)))
