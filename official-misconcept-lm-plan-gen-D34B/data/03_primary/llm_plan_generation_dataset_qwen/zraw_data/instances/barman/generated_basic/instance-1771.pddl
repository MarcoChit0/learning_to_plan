(define (problem prob)
 (:domain barman)
 (:objects 
      shaker225 - shaker
      left right - hand
      shot8 shot151 shot457 - shot
      ingredient47 ingredient398 - ingredient
      cocktail204 - cocktail
      dispenser447 dispenser37 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker225)
  (ontable shot8)
  (ontable shot151)
  (ontable shot457)
  (dispenses dispenser447 ingredient47)
  (dispenses dispenser37 ingredient398)
  (clean shaker225)
  (clean shot8)
  (clean shot151)
  (clean shot457)
  (empty shaker225)
  (empty shot8)
  (empty shot151)
  (empty shot457)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker225 l0)
  (shaker-level shaker225 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail204 ingredient47)
  (cocktail-part2 cocktail204 ingredient398)
)
 (:goal
  (and
      (contains shot8 cocktail204)
      (contains shot151 cocktail204)
)))
