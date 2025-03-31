(define (problem prob)
 (:domain barman)
 (:objects 
      shaker7 - shaker
      left right - hand
      shot172 shot199 - shot
      ingredient333 ingredient296 ingredient187 ingredient241 - ingredient
      cocktail58 - cocktail
      dispenser219 dispenser376 dispenser12 dispenser109 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker7)
  (ontable shot172)
  (ontable shot199)
  (dispenses dispenser219 ingredient333)
  (dispenses dispenser376 ingredient296)
  (dispenses dispenser12 ingredient187)
  (dispenses dispenser109 ingredient241)
  (clean shaker7)
  (clean shot172)
  (clean shot199)
  (empty shaker7)
  (empty shot172)
  (empty shot199)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker7 l0)
  (shaker-level shaker7 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail58 ingredient241)
  (cocktail-part2 cocktail58 ingredient187)
)
 (:goal
  (and
      (contains shot172 cocktail58)
)))
