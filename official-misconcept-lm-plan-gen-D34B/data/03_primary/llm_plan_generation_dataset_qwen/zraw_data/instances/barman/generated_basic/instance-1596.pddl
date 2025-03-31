(define (problem prob)
 (:domain barman)
 (:objects 
      shaker191 - shaker
      left right - hand
      shot92 shot187 - shot
      ingredient137 ingredient147 - ingredient
      cocktail236 - cocktail
      dispenser145 dispenser403 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker191)
  (ontable shot92)
  (ontable shot187)
  (dispenses dispenser145 ingredient137)
  (dispenses dispenser403 ingredient147)
  (clean shaker191)
  (clean shot92)
  (clean shot187)
  (empty shaker191)
  (empty shot92)
  (empty shot187)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker191 l0)
  (shaker-level shaker191 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail236 ingredient147)
  (cocktail-part2 cocktail236 ingredient137)
)
 (:goal
  (and
      (contains shot92 cocktail236)
)))
