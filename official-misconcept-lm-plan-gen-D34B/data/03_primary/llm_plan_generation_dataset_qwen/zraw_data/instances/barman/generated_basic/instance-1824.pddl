(define (problem prob)
 (:domain barman)
 (:objects 
      shaker329 - shaker
      left right - hand
      shot42 shot33 shot89 - shot
      ingredient363 ingredient245 ingredient68 ingredient440 - ingredient
      cocktail386 - cocktail
      dispenser334 dispenser177 dispenser87 dispenser474 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker329)
  (ontable shot42)
  (ontable shot33)
  (ontable shot89)
  (dispenses dispenser334 ingredient363)
  (dispenses dispenser177 ingredient245)
  (dispenses dispenser87 ingredient68)
  (dispenses dispenser474 ingredient440)
  (clean shaker329)
  (clean shot42)
  (clean shot33)
  (clean shot89)
  (empty shaker329)
  (empty shot42)
  (empty shot33)
  (empty shot89)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker329 l0)
  (shaker-level shaker329 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail386 ingredient440)
  (cocktail-part2 cocktail386 ingredient68)
)
 (:goal
  (and
      (contains shot42 cocktail386)
      (contains shot33 ingredient363)
)))
