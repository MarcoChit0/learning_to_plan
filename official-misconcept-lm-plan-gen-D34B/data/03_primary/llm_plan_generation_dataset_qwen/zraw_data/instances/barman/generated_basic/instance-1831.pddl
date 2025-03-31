(define (problem prob)
 (:domain barman)
 (:objects 
      shaker221 - shaker
      left right - hand
      shot209 shot419 shot359 - shot
      ingredient36 ingredient48 ingredient337 ingredient106 - ingredient
      cocktail238 - cocktail
      dispenser86 dispenser334 dispenser109 dispenser254 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker221)
  (ontable shot209)
  (ontable shot419)
  (ontable shot359)
  (dispenses dispenser86 ingredient36)
  (dispenses dispenser334 ingredient48)
  (dispenses dispenser109 ingredient337)
  (dispenses dispenser254 ingredient106)
  (clean shaker221)
  (clean shot209)
  (clean shot419)
  (clean shot359)
  (empty shaker221)
  (empty shot209)
  (empty shot419)
  (empty shot359)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker221 l0)
  (shaker-level shaker221 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail238 ingredient48)
  (cocktail-part2 cocktail238 ingredient337)
)
 (:goal
  (and
      (contains shot209 cocktail238)
      (contains shot419 cocktail238)
)))
