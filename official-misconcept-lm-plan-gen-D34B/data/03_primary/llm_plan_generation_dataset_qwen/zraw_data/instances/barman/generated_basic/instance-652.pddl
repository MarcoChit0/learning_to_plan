(define (problem prob)
 (:domain barman)
 (:objects 
      shaker469 - shaker
      left right - hand
      shot386 shot268 - shot
      ingredient357 ingredient195 ingredient63 - ingredient
      cocktail1 - cocktail
      dispenser446 dispenser44 dispenser102 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker469)
  (ontable shot386)
  (ontable shot268)
  (dispenses dispenser446 ingredient357)
  (dispenses dispenser44 ingredient195)
  (dispenses dispenser102 ingredient63)
  (clean shaker469)
  (clean shot386)
  (clean shot268)
  (empty shaker469)
  (empty shot386)
  (empty shot268)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker469 l0)
  (shaker-level shaker469 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient357)
  (cocktail-part2 cocktail1 ingredient195)
)
 (:goal
  (and
      (contains shot386 cocktail1)
)))
