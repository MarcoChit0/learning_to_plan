(define (problem prob)
 (:domain barman)
 (:objects 
      shaker25 - shaker
      left right - hand
      shot260 shot195 - shot
      ingredient461 ingredient152 ingredient359 ingredient187 - ingredient
      cocktail188 - cocktail
      dispenser453 dispenser33 dispenser243 dispenser233 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker25)
  (ontable shot260)
  (ontable shot195)
  (dispenses dispenser453 ingredient461)
  (dispenses dispenser33 ingredient152)
  (dispenses dispenser243 ingredient359)
  (dispenses dispenser233 ingredient187)
  (clean shaker25)
  (clean shot260)
  (clean shot195)
  (empty shaker25)
  (empty shot260)
  (empty shot195)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker25 l0)
  (shaker-level shaker25 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail188 ingredient359)
  (cocktail-part2 cocktail188 ingredient152)
)
 (:goal
  (and
      (contains shot260 cocktail188)
)))
