(define (problem prob)
 (:domain barman)
 (:objects 
      shaker315 - shaker
      left right - hand
      shot236 shot172 - shot
      ingredient463 ingredient406 ingredient196 ingredient95 - ingredient
      cocktail140 - cocktail
      dispenser178 dispenser209 dispenser230 dispenser88 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker315)
  (ontable shot236)
  (ontable shot172)
  (dispenses dispenser178 ingredient463)
  (dispenses dispenser209 ingredient406)
  (dispenses dispenser230 ingredient196)
  (dispenses dispenser88 ingredient95)
  (clean shaker315)
  (clean shot236)
  (clean shot172)
  (empty shaker315)
  (empty shot236)
  (empty shot172)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker315 l0)
  (shaker-level shaker315 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail140 ingredient95)
  (cocktail-part2 cocktail140 ingredient196)
)
 (:goal
  (and
      (contains shot236 cocktail140)
)))
