(define (problem prob)
 (:domain barman)
 (:objects 
      shaker360 - shaker
      left right - hand
      shot89 shot256 shot137 - shot
      ingredient440 ingredient300 ingredient233 ingredient369 - ingredient
      cocktail169 - cocktail
      dispenser69 dispenser113 dispenser213 dispenser60 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker360)
  (ontable shot89)
  (ontable shot256)
  (ontable shot137)
  (dispenses dispenser69 ingredient440)
  (dispenses dispenser113 ingredient300)
  (dispenses dispenser213 ingredient233)
  (dispenses dispenser60 ingredient369)
  (clean shaker360)
  (clean shot89)
  (clean shot256)
  (clean shot137)
  (empty shaker360)
  (empty shot89)
  (empty shot256)
  (empty shot137)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker360 l0)
  (shaker-level shaker360 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail169 ingredient369)
  (cocktail-part2 cocktail169 ingredient233)
)
 (:goal
  (and
      (contains shot89 cocktail169)
      (contains shot256 cocktail169)
)))
