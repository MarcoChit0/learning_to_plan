(define (problem prob)
 (:domain barman)
 (:objects 
      shaker447 - shaker
      left right - hand
      shot155 shot38 - shot
      ingredient372 ingredient462 ingredient174 ingredient337 - ingredient
      cocktail464 - cocktail
      dispenser178 dispenser217 dispenser286 dispenser403 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker447)
  (ontable shot155)
  (ontable shot38)
  (dispenses dispenser178 ingredient372)
  (dispenses dispenser217 ingredient462)
  (dispenses dispenser286 ingredient174)
  (dispenses dispenser403 ingredient337)
  (clean shaker447)
  (clean shot155)
  (clean shot38)
  (empty shaker447)
  (empty shot155)
  (empty shot38)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker447 l0)
  (shaker-level shaker447 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail464 ingredient174)
  (cocktail-part2 cocktail464 ingredient462)
)
 (:goal
  (and
      (contains shot155 cocktail464)
)))
