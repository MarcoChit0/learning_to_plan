(define (problem prob)
 (:domain barman)
 (:objects 
      shaker498 - shaker
      left right - hand
      shot332 - shot
      ingredient362 ingredient129 ingredient199 ingredient397 - ingredient
      cocktail248 - cocktail
      dispenser150 dispenser149 dispenser163 dispenser391 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker498)
  (ontable shot332)
  (dispenses dispenser150 ingredient362)
  (dispenses dispenser149 ingredient129)
  (dispenses dispenser163 ingredient199)
  (dispenses dispenser391 ingredient397)
  (clean shaker498)
  (clean shot332)
  (empty shaker498)
  (empty shot332)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker498 l0)
  (shaker-level shaker498 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail248 ingredient199)
  (cocktail-part2 cocktail248 ingredient129)
)
 (:goal
  (and
      (contains shot332 cocktail248)
)))
