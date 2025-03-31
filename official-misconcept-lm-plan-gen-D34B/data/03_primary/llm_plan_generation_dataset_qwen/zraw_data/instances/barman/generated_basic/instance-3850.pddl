(define (problem prob)
 (:domain barman)
 (:objects 
      shaker389 - shaker
      left right - hand
      shot187 shot81 shot77 - shot
      ingredient144 ingredient168 - ingredient
      cocktail204 - cocktail
      dispenser117 dispenser68 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker389)
  (ontable shot187)
  (ontable shot81)
  (ontable shot77)
  (dispenses dispenser117 ingredient144)
  (dispenses dispenser68 ingredient168)
  (clean shaker389)
  (clean shot187)
  (clean shot81)
  (clean shot77)
  (empty shaker389)
  (empty shot187)
  (empty shot81)
  (empty shot77)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker389 l0)
  (shaker-level shaker389 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail204 ingredient168)
  (cocktail-part2 cocktail204 ingredient144)
)
 (:goal
  (and
      (contains shot187 cocktail204)
      (contains shot81 cocktail204)
)))
