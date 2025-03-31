(define (problem prob)
 (:domain barman)
 (:objects 
      shaker117 - shaker
      left right - hand
      shot425 shot81 shot234 - shot
      ingredient129 ingredient112 ingredient396 ingredient268 - ingredient
      cocktail126 - cocktail
      dispenser310 dispenser457 dispenser223 dispenser143 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker117)
  (ontable shot425)
  (ontable shot81)
  (ontable shot234)
  (dispenses dispenser310 ingredient129)
  (dispenses dispenser457 ingredient112)
  (dispenses dispenser223 ingredient396)
  (dispenses dispenser143 ingredient268)
  (clean shaker117)
  (clean shot425)
  (clean shot81)
  (clean shot234)
  (empty shaker117)
  (empty shot425)
  (empty shot81)
  (empty shot234)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker117 l0)
  (shaker-level shaker117 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail126 ingredient396)
  (cocktail-part2 cocktail126 ingredient268)
)
 (:goal
  (and
      (contains shot425 cocktail126)
      (contains shot81 ingredient129)
)))
