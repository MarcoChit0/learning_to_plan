(define (problem prob)
 (:domain barman)
 (:objects 
      shaker331 - shaker
      left right - hand
      shot312 shot141 - shot
      ingredient406 ingredient340 - ingredient
      cocktail256 - cocktail
      dispenser453 dispenser407 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker331)
  (ontable shot312)
  (ontable shot141)
  (dispenses dispenser453 ingredient406)
  (dispenses dispenser407 ingredient340)
  (clean shaker331)
  (clean shot312)
  (clean shot141)
  (empty shaker331)
  (empty shot312)
  (empty shot141)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker331 l0)
  (shaker-level shaker331 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail256 ingredient406)
  (cocktail-part2 cocktail256 ingredient340)
)
 (:goal
  (and
      (contains shot312 cocktail256)
)))
