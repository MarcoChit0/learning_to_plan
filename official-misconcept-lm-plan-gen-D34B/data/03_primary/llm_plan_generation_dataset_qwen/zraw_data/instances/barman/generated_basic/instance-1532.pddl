(define (problem prob)
 (:domain barman)
 (:objects 
      shaker146 - shaker
      left right - hand
      shot456 shot137 - shot
      ingredient88 ingredient182 ingredient155 ingredient196 - ingredient
      cocktail432 - cocktail
      dispenser97 dispenser148 dispenser59 dispenser239 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker146)
  (ontable shot456)
  (ontable shot137)
  (dispenses dispenser97 ingredient88)
  (dispenses dispenser148 ingredient182)
  (dispenses dispenser59 ingredient155)
  (dispenses dispenser239 ingredient196)
  (clean shaker146)
  (clean shot456)
  (clean shot137)
  (empty shaker146)
  (empty shot456)
  (empty shot137)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker146 l0)
  (shaker-level shaker146 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail432 ingredient88)
  (cocktail-part2 cocktail432 ingredient182)
)
 (:goal
  (and
      (contains shot456 cocktail432)
)))
