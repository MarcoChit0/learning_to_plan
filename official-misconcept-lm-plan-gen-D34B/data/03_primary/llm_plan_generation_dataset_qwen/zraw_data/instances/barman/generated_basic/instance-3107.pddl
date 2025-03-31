(define (problem prob)
 (:domain barman)
 (:objects 
      shaker41 - shaker
      left right - hand
      shot455 shot456 - shot
      ingredient179 ingredient354 - ingredient
      cocktail385 - cocktail
      dispenser109 dispenser478 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker41)
  (ontable shot455)
  (ontable shot456)
  (dispenses dispenser109 ingredient179)
  (dispenses dispenser478 ingredient354)
  (clean shaker41)
  (clean shot455)
  (clean shot456)
  (empty shaker41)
  (empty shot455)
  (empty shot456)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker41 l0)
  (shaker-level shaker41 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail385 ingredient179)
  (cocktail-part2 cocktail385 ingredient354)
)
 (:goal
  (and
      (contains shot455 cocktail385)
)))
