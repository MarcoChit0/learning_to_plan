(define (problem prob)
 (:domain barman)
 (:objects 
      shaker413 - shaker
      left right - hand
      shot131 shot383 shot389 - shot
      ingredient43 ingredient188 ingredient323 ingredient79 - ingredient
      cocktail403 - cocktail
      dispenser157 dispenser206 dispenser457 dispenser385 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker413)
  (ontable shot131)
  (ontable shot383)
  (ontable shot389)
  (dispenses dispenser157 ingredient43)
  (dispenses dispenser206 ingredient188)
  (dispenses dispenser457 ingredient323)
  (dispenses dispenser385 ingredient79)
  (clean shaker413)
  (clean shot131)
  (clean shot383)
  (clean shot389)
  (empty shaker413)
  (empty shot131)
  (empty shot383)
  (empty shot389)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker413 l0)
  (shaker-level shaker413 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail403 ingredient43)
  (cocktail-part2 cocktail403 ingredient323)
)
 (:goal
  (and
      (contains shot131 cocktail403)
      (contains shot383 ingredient79)
)))
