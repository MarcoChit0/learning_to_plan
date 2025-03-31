(define (problem prob)
 (:domain barman)
 (:objects 
      shaker317 - shaker
      left right - hand
      shot140 shot191 - shot
      ingredient353 ingredient62 ingredient82 - ingredient
      cocktail220 - cocktail
      dispenser174 dispenser53 dispenser390 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker317)
  (ontable shot140)
  (ontable shot191)
  (dispenses dispenser174 ingredient353)
  (dispenses dispenser53 ingredient62)
  (dispenses dispenser390 ingredient82)
  (clean shaker317)
  (clean shot140)
  (clean shot191)
  (empty shaker317)
  (empty shot140)
  (empty shot191)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker317 l0)
  (shaker-level shaker317 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail220 ingredient82)
  (cocktail-part2 cocktail220 ingredient353)
)
 (:goal
  (and
      (contains shot140 cocktail220)
)))
