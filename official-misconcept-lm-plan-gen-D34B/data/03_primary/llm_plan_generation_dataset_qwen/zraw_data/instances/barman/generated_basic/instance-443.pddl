(define (problem prob)
 (:domain barman)
 (:objects 
      shaker229 - shaker
      left right - hand
      shot11 shot125 - shot
      ingredient482 ingredient360 ingredient422 - ingredient
      cocktail1 - cocktail
      dispenser245 dispenser256 dispenser455 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker229)
  (ontable shot11)
  (ontable shot125)
  (dispenses dispenser245 ingredient482)
  (dispenses dispenser256 ingredient360)
  (dispenses dispenser455 ingredient422)
  (clean shaker229)
  (clean shot11)
  (clean shot125)
  (empty shaker229)
  (empty shot11)
  (empty shot125)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker229 l0)
  (shaker-level shaker229 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient360)
  (cocktail-part2 cocktail1 ingredient422)
)
 (:goal
  (and
      (contains shot11 cocktail1)
)))
