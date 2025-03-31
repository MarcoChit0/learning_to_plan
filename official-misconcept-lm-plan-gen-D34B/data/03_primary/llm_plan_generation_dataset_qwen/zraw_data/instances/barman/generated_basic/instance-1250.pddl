(define (problem prob)
 (:domain barman)
 (:objects 
      shaker385 - shaker
      left right - hand
      shot129 shot404 shot186 - shot
      ingredient280 ingredient384 ingredient338 ingredient286 - ingredient
      cocktail124 - cocktail
      dispenser290 dispenser328 dispenser88 dispenser142 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker385)
  (ontable shot129)
  (ontable shot404)
  (ontable shot186)
  (dispenses dispenser290 ingredient280)
  (dispenses dispenser328 ingredient384)
  (dispenses dispenser88 ingredient338)
  (dispenses dispenser142 ingredient286)
  (clean shaker385)
  (clean shot129)
  (clean shot404)
  (clean shot186)
  (empty shaker385)
  (empty shot129)
  (empty shot404)
  (empty shot186)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker385 l0)
  (shaker-level shaker385 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail124 ingredient338)
  (cocktail-part2 cocktail124 ingredient280)
)
 (:goal
  (and
      (contains shot129 cocktail124)
      (contains shot404 ingredient338)
)))
