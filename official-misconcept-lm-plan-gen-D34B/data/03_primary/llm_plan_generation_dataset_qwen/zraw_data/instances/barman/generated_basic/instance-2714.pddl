(define (problem prob)
 (:domain barman)
 (:objects 
      shaker295 - shaker
      left right - hand
      shot249 shot400 shot374 - shot
      ingredient425 ingredient263 - ingredient
      cocktail422 - cocktail
      dispenser135 dispenser247 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker295)
  (ontable shot249)
  (ontable shot400)
  (ontable shot374)
  (dispenses dispenser135 ingredient425)
  (dispenses dispenser247 ingredient263)
  (clean shaker295)
  (clean shot249)
  (clean shot400)
  (clean shot374)
  (empty shaker295)
  (empty shot249)
  (empty shot400)
  (empty shot374)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker295 l0)
  (shaker-level shaker295 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail422 ingredient263)
  (cocktail-part2 cocktail422 ingredient425)
)
 (:goal
  (and
      (contains shot249 cocktail422)
      (contains shot400 cocktail422)
)))
