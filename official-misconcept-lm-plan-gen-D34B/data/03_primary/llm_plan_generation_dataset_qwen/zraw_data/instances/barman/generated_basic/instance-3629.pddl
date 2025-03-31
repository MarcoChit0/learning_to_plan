(define (problem prob)
 (:domain barman)
 (:objects 
      shaker305 - shaker
      left right - hand
      shot200 - shot
      ingredient26 ingredient435 ingredient392 ingredient449 - ingredient
      cocktail341 - cocktail
      dispenser418 dispenser125 dispenser227 dispenser217 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker305)
  (ontable shot200)
  (dispenses dispenser418 ingredient26)
  (dispenses dispenser125 ingredient435)
  (dispenses dispenser227 ingredient392)
  (dispenses dispenser217 ingredient449)
  (clean shaker305)
  (clean shot200)
  (empty shaker305)
  (empty shot200)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker305 l0)
  (shaker-level shaker305 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail341 ingredient435)
  (cocktail-part2 cocktail341 ingredient26)
)
 (:goal
  (and
      (contains shot200 cocktail341)
)))
