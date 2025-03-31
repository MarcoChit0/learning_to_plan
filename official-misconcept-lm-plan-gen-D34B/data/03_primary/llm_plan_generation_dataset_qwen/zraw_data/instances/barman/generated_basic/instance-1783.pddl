(define (problem prob)
 (:domain barman)
 (:objects 
      shaker489 - shaker
      left right - hand
      shot434 shot260 - shot
      ingredient153 ingredient71 ingredient144 - ingredient
      cocktail14 - cocktail
      dispenser6 dispenser139 dispenser193 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker489)
  (ontable shot434)
  (ontable shot260)
  (dispenses dispenser6 ingredient153)
  (dispenses dispenser139 ingredient71)
  (dispenses dispenser193 ingredient144)
  (clean shaker489)
  (clean shot434)
  (clean shot260)
  (empty shaker489)
  (empty shot434)
  (empty shot260)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker489 l0)
  (shaker-level shaker489 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail14 ingredient144)
  (cocktail-part2 cocktail14 ingredient71)
)
 (:goal
  (and
      (contains shot434 cocktail14)
)))
